import os
import json
import time
import logging
import torch
import pandas as pd
from tqdm import tqdm
from evaluation.metrics import full_evaluation
from src.model import load_model_and_processor
from src.conversation import build_conversation, build_conversation_mcq
from src.inference import get_model_output
from src.utils import validate_audio_path, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Full benchmark runner for Audio Temporal Reasoning Pipeline.

    Runs inference on a test set, computes all evaluation metrics,
    measures latency, and saves a detailed report.

    Usage:
        runner = BenchmarkRunner(
            model_id="Qwen/Qwen2.5-Omni-7B",
            lora_path="checkpoints/final_adapter",
            audio_root="data/audio/",
            test_json="data/test.json",
            output_dir="outputs/"
        )
        results = runner.run()
    """

    def __init__(
        self,
        model_id:    str,
        audio_root:  str,
        test_json:   str,
        output_dir:  str  = "outputs/",
        lora_path:   str  = None,
        gpu_id:      int  = 0,
        task:        str  = "temporal",
        seed:        int  = 42,
        use_4bit:    bool = False,
    ):
        self.model_id   = model_id
        self.audio_root = audio_root
        self.test_json  = test_json
        self.output_dir = output_dir
        self.lora_path  = lora_path
        self.gpu_id     = gpu_id
        self.task       = task

        os.makedirs(output_dir, exist_ok=True)
        set_seed(seed)

        # Load model
        logger.info("Loading model for benchmarking...")
        self.model, self.processor = load_model_and_processor(
            model_id=model_id,
            lora_path=lora_path,
            gpu_id=gpu_id,
            use_4bit=use_4bit,
        )

        # Load test data
        with open(test_json, 'r') as f:
            self.samples = json.load(f)

        logger.info(f"Loaded {len(self.samples)} test samples")

    def run(self) -> dict:
        """
        Run full benchmark:
        - Inference on all test samples
        - Latency tracking per sample
        - Metric computation
        - Report generation
        """
        logger.info("Starting benchmark run...")

        predictions  = []
        references   = []
        latencies    = []
        outputs      = []

        for sample in tqdm(self.samples, desc="Benchmarking"):
            audio_path = os.path.join(
                self.audio_root, sample["audio"] + ".wav"
            )

            # Skip missing audio
            if not validate_audio_path(audio_path):
                sample["model_prediction"] = "AUDIO_NOT_FOUND"
                sample["latency_ms"]       = -1
                outputs.append(sample)
                continue

            # Build conversation
            if "choices" in sample:
                conversation = build_conversation_mcq(
                    self.processor,
                    sample["prompt"],
                    sample["choices"],
                    audio_path
                )
            else:
                conversation = build_conversation(
                    self.processor,
                    sample["prompt"],
                    audio_path
                )

            # Timed inference
            start_time = time.perf_counter()
            pred       = get_model_output(self.model, self.processor, conversation)
            end_time   = time.perf_counter()

            latency_ms = round((end_time - start_time) * 1000, 2)

            sample["model_prediction"] = pred
            sample["latency_ms"]       = latency_ms

            predictions.append(pred)
            latencies.append(latency_ms)
            outputs.append(sample)

            if "answer" in sample:
                references.append(sample["answer"])

        # Compute metrics
        metrics = {}
        if references and len(references) == len(predictions):
            metrics = full_evaluation(predictions, references, task=self.task)

        # Latency stats
        if latencies:
            sorted_lat = sorted(latencies)
            n          = len(sorted_lat)
            metrics["latency_p50_ms"] = round(sorted_lat[int(n * 0.50)], 2)
            metrics["latency_p95_ms"] = round(sorted_lat[int(n * 0.95)], 2)
            metrics["latency_mean_ms"] = round(sum(latencies) / n, 2)
            metrics["latency_max_ms"]  = round(max(latencies), 2)

        metrics["total_samples"]  = len(self.samples)
        metrics["skipped_samples"] = sum(
            1 for o in outputs if o["model_prediction"] == "AUDIO_NOT_FOUND"
        )

        # Save outputs
        pred_path   = os.path.join(self.output_dir, "predictions.json")
        report_path = os.path.join(self.output_dir, "benchmark_report.json")

        with open(pred_path, 'w') as f:
            json.dump(outputs, f, indent=2)

        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Print report
        self._print_report(metrics)

        return metrics

    def _print_report(self, metrics: dict):
        """Print formatted benchmark report."""
        print("\n" + "=" * 55)
        print("  BENCHMARK REPORT")
        print("=" * 55)
        print(f"  Model          : {self.model_id}")
        print(f"  LoRA adapter   : {self.lora_path or 'None'}")
        print(f"  Total samples  : {metrics.get('total_samples', 'N/A')}")
        print(f"  Skipped        : {metrics.get('skipped_samples', 0)}")
        print("-" * 55)

        if "exact_match" in metrics:
            print(f"  Exact match    : {metrics['exact_match']}%")
        if "rouge_1" in metrics:
            print(f"  ROUGE-1        : {metrics['rouge_1']}")
        if "rouge_l" in metrics:
            print(f"  ROUGE-L        : {metrics['rouge_l']}")
        if "bert_score" in metrics:
            print(f"  BERTScore F1   : {metrics['bert_score']}")
        if "hallucination_rate" in metrics:
            print(f"  Hallucination  : {metrics['hallucination_rate']}%")
        if "temporal_ordering_accuracy" in metrics:
            print(f"  Temporal Order : {metrics['temporal_ordering_accuracy']}%")
        if "sound_event_recall" in metrics:
            print(f"  Event Recall   : {metrics['sound_event_recall']}%")

        print("-" * 55)
        print(f"  Latency p50    : {metrics.get('latency_p50_ms', 'N/A')} ms")
        print(f"  Latency p95    : {metrics.get('latency_p95_ms', 'N/A')} ms")
        print(f"  Latency mean   : {metrics.get('latency_mean_ms', 'N/A')} ms")
        print(f"  Latency max    : {metrics.get('latency_max_ms', 'N/A')} ms")
        print("=" * 55 + "\n")
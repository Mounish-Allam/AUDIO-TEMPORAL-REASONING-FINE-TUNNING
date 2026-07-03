import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Windows defaults stdout/stderr to the system codepage (not UTF-8) when
# they aren't attached to a real console (e.g. redirected to a file or a
# background process) — this print module uses non-ASCII characters
# (arrows, em-dashes), which would otherwise raise UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
from tqdm import tqdm
from src.model import load_model_and_processor
from src.conversation import build_conversation, build_conversation_mcq
from src.inference import get_model_output
from src.utils import validate_audio_path, load_json, save_json, set_seed
from evaluation.metrics import full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Batch audio inference with Qwen2.5-Omni")
    parser.add_argument("--audio_root", type=str, required=True)
    parser.add_argument("--test_json",  type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    parser.add_argument("--model_id",   type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--lora_path",  type=str, default=None)
    parser.add_argument("--gpu_id",     type=int, default=0)
    parser.add_argument("--lora",       action="store_true")
    parser.add_argument("--use_4bit",   action="store_true",
                        help="Load model in 4-bit (fits a free 16GB T4 GPU)")
    parser.add_argument("--task",       type=str, default="temporal",
                        choices=["temporal", "open", "mcq"],
                        help="Evaluation task type (default: temporal)")
    args = parser.parse_args()

    if args.lora and not args.lora_path:
        raise ValueError("--lora_path must be provided when --lora flag is set")

    set_seed(42)

    model, processor = load_model_and_processor(
        args.model_id,
        lora_path=args.lora_path if args.lora else None,
        gpu_id=args.gpu_id,
        use_4bit=args.use_4bit,
    )

    samples  = load_json(args.test_json)
    outputs  = []
    latencies = []
    predictions = []
    references  = []

    for sample in tqdm(samples, desc="Running inference"):
        audio_path = os.path.join(args.audio_root, sample["audio"] + ".wav")

        if not validate_audio_path(audio_path):
            sample["model_prediction"] = "AUDIO_NOT_FOUND"
            sample["latency_ms"]       = -1
            outputs.append(sample)
            continue

        if "choices" in sample:
            conversation = build_conversation_mcq(
                processor, sample["prompt"], sample["choices"], audio_path
            )
        else:
            conversation = build_conversation(processor, sample["prompt"], audio_path)

        t0   = time.perf_counter()
        pred = get_model_output(model, processor, conversation)
        t1   = time.perf_counter()

        latency_ms = round((t1 - t0) * 1000, 1)
        sample["model_prediction"] = pred
        sample["latency_ms"]       = latency_ms

        latencies.append(latency_ms)
        outputs.append(sample)

        if "answer" in sample:
            predictions.append(pred)
            references.append(sample["answer"])

    save_json(outputs, args.output)

    # Print summary
    total   = len(samples)
    skipped = sum(1 for o in outputs if o["model_prediction"] == "AUDIO_NOT_FOUND")
    print(f"\n  Saved {total} predictions → {args.output}  ({skipped} skipped)")

    metrics = {}
    if predictions and references:
        metrics = full_evaluation(predictions, references, task=args.task)
        print("\n  ── Evaluation metrics ───────────────────")
        for k, v in metrics.items():
            print(f"  {k:<32} {v:.1f}%")

    if latencies:
        s = sorted(latencies)
        n = len(s)
        metrics["latency_p50_ms"]  = round(s[int(n * 0.50)], 2)
        metrics["latency_p95_ms"]  = round(s[int(n * 0.95)], 2)
        metrics["latency_mean_ms"] = round(sum(latencies) / n, 2)
        metrics["latency_max_ms"]  = round(max(latencies), 2)
        print(f"\n  ── Latency ──────────────────────────────")
        print(f"  p50  : {s[int(n*0.50)]:.0f} ms")
        print(f"  p95  : {s[int(n*0.95)]:.0f} ms")
        print(f"  mean : {sum(latencies)/n:.0f} ms")

    metrics["total_samples"]   = total
    metrics["skipped_samples"] = skipped

    # Written alongside --output so the Streamlit dashboard (which reads a
    # fixed "outputs/benchmark_report.json") always has metrics to show —
    # matches the file evaluation/benchmark.py's BenchmarkRunner produces.
    report_path = os.path.join(os.path.dirname(args.output) or ".", "benchmark_report.json")
    save_json(metrics, report_path)
    print(f"\n  Saved report → {report_path}")


if __name__ == "__main__":
    main()
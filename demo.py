"""
Dry-run demo for the Audio Temporal Reasoning Pipeline.

Validates every module in the pipeline using lightweight mock objects —
no GPU, no model download required.  Run this first to confirm the
environment is set up correctly before using real model weights.

Usage:
    python demo.py
"""
import os
import sys
import json
import numpy as np
from unittest.mock import MagicMock, patch

# Windows defaults stdout/stderr to the system codepage (not UTF-8) when
# they aren't attached to a real console (e.g. redirected to a file) —
# this file's prints use non-ASCII characters (arrows, em-dashes).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))


class SkipStep(Exception):
    """Raised when a step's inputs don't exist yet (not a failure)."""


def _make_mock_processor():
    processor = MagicMock()
    processor.feature_extractor.sampling_rate = 16_000

    # apply_chat_template returns a formatted prompt string
    processor.apply_chat_template.return_value = (
        "<|im_start|>system\nYou are Qwen.<|im_end|>\n"
        "<|im_start|>user\n<audio>What do you hear?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # __call__ (tokenise + process audio) returns an inputs-like object
    mock_inputs = MagicMock()
    mock_inputs.input_ids.shape = (1, 42)
    mock_inputs.to.return_value = mock_inputs
    processor.return_value = mock_inputs

    # batch_decode returns a list of decoded strings
    processor.batch_decode.return_value = [
        "The audio begins with a dog barking, followed by rain."
    ]
    processor.tokenizer = MagicMock()
    return processor


def _make_mock_model(device="cpu"):
    model = MagicMock()
    model.device = device
    # generate returns a token tensor shaped (1, 50) — longer than input (42)
    import torch
    model.generate.return_value = torch.zeros(1, 50, dtype=torch.long)
    return model


# ── Demo steps ────────────────────────────────────────────────────────────────

def demo_utils():
    print("  [1/6] Utils …")
    from src.utils import validate_audio_path, load_json, save_json, set_seed
    set_seed(42)
    assert not validate_audio_path("nonexistent.wav")
    # save / load round-trip
    tmp = "outputs/_demo_tmp.json"
    save_json({"ok": True}, tmp)
    data = load_json(tmp)
    assert data == {"ok": True}
    os.remove(tmp)
    print("        PASS — utils work correctly")


def demo_conversation():
    print("  [2/6] Conversation builder …")
    processor = _make_mock_processor()

    with patch("qwen_omni_utils.process_mm_info", return_value=([np.zeros(16000)], [], [])):
        from src.conversation import build_conversation, build_conversation_mcq

        conv = build_conversation(processor, "What sound do you hear?", "fake.wav")
        assert "prompt" in conv and "audios" in conv

        mcq = build_conversation_mcq(
            processor,
            "Which sound is dominant?",
            ["dog bark", "rain", "music", "speech"],
            "fake.wav"
        )
        assert "prompt" in mcq and "audios" in mcq
    print("        PASS — conversation builders work correctly")


def demo_inference():
    print("  [3/6] Inference pipeline …")
    import torch
    model     = _make_mock_model()
    processor = _make_mock_processor()

    from src.inference import get_model_output
    conversation = {
        "prompt": "<|im_start|>user\nWhat do you hear?<|im_end|>",
        "audios": [np.zeros(16_000, dtype=np.float32)]
    }
    result = get_model_output(model, processor, conversation)
    assert isinstance(result, str) and len(result) > 0
    print(f"        PASS — inference returned: \"{result[:60]}...\"")


def demo_metrics():
    print("  [4/6] Evaluation metrics …")
    from evaluation.metrics import (
        compute_exact_match, compute_rouge_l,
        hallucination_rate, full_evaluation
    )

    preds = ["car engine", "dog barking then rain", "three events"]
    refs  = ["car engine", "dog barking, then rain", "three"]

    em   = compute_exact_match(preds, refs)
    rl   = compute_rouge_l(preds, refs)
    hr   = hallucination_rate(preds, refs)
    full = full_evaluation(preds, refs, task="open")

    assert 0 <= em  <= 100
    assert 0 <= rl  <= 100
    assert 0 <= hr  <= 100
    assert "rouge_l" in full
    print(f"        PASS — EM={em:.1f}%  ROUGE-L={rl:.1f}  Hallucination={hr:.1f}%")


def demo_data():
    print("  [5/6] Data files …")
    from src.utils import load_json
    import os

    # These only exist after running scripts/prepare_audiocaps.py —
    # their absence is expected on a fresh clone, so we skip, not fail.
    train_path = "data/train.json"
    test_path  = "data/test.json"
    pref_path  = "data/preference_pairs.json"

    if not all(os.path.exists(p) for p in (train_path, test_path, pref_path)):
        raise SkipStep(
            "no dataset yet — run: python scripts/prepare_audiocaps.py"
        )

    train = load_json(train_path)
    test  = load_json(test_path)
    prefs = load_json(pref_path)

    assert train and len(train) >= 1, "train.json is empty"
    assert test  and len(test)  >= 1, "test.json is empty"
    assert prefs and len(prefs) >= 1, "preference_pairs.json is empty"
    assert "answer"   in train[0], "train sample missing 'answer' key"
    assert "chosen"   in prefs[0], "preference pair missing 'chosen' key"
    assert "rejected" in prefs[0], "preference pair missing 'rejected' key"
    print(f"        PASS — {len(train)} train / {len(test)} test / {len(prefs)} pref pairs (AudioCaps)")


def demo_outputs():
    print("  [6/6] Benchmark outputs …")
    from src.utils import load_json

    # Only exist after a real evaluation run on a GPU — skip, not fail.
    if not (os.path.exists("outputs/benchmark_report.json")
            and os.path.exists("outputs/predictions.json")):
        raise SkipStep(
            "no benchmark yet — run: python scripts/run_batch_inference.py (GPU)"
        )

    report = load_json("outputs/benchmark_report.json")
    preds  = load_json("outputs/predictions.json")

    assert "rouge_l" in report and "hallucination_rate" in report
    if any(p.get("simulated") for p in preds):
        print(f"        PASS — but predictions are SIMULATED (fake demo data)")
    else:
        print(f"        PASS — report has {len(report)} metrics, {len(preds)} predictions")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Audio Temporal Reasoning Pipeline — dry-run demo")
    print("=" * 60)

    steps = [
        demo_utils,
        demo_conversation,
        demo_inference,
        demo_metrics,
        demo_data,
        demo_outputs,
    ]

    passed  = 0
    skipped = 0
    for step in steps:
        try:
            step()
            passed += 1
        except SkipStep as why:
            skipped += 1
            print(f"        SKIP — {why}")
        except Exception as exc:
            print(f"        FAIL — {exc}")

    print("=" * 60)
    print(f"  {passed}/{len(steps)} checks passed"
          + (f"  ({skipped} skipped — see messages above)" if skipped else ""))
    if passed + skipped == len(steps):
        print("  All systems nominal.")
        print("\n  Next steps:")
        print("    python scripts/prepare_audiocaps.py      # download AudioCaps dataset")
        print("    python scripts/run_finetune.py           # QLoRA SFT training")
        print("    python scripts/run_dpo.py                # DPO preference optimization")
        print("    streamlit run dashboard/app.py           # launch results dashboard")
        print("    pytest tests/ -v                         # run unit tests")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

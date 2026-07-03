"""
*** FAKE DATA GENERATOR — FOR DASHBOARD LAYOUT TESTING ONLY ***

This script creates SIMULATED predictions by mangling the ground-truth
captions. No model is run. The numbers it produces are MEANINGLESS as
evaluation results and must NEVER be quoted in the README, a resume,
or an interview.

Every record it writes is stamped with  "simulated": true  so the
dashboard can show a warning banner. Real predictions come from:

    python scripts/run_batch_inference.py ...   (needs a GPU)

Usage (dashboard layout testing only):
    python scripts/make_fake_dashboard_data.py --max_samples 200 --stage dpo
"""

import argparse
import json
import os
import sys
import random
import re
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TEMPORAL_PROMPTS = [
    "Describe the temporal order of events in this audio clip.",
    "What sequence of sounds do you hear in this recording? Describe them in order.",
    "List the sound events in this audio from first to last.",
    "In what order do the sounds occur in this audio clip?",
    "Describe what happens first, then next, and finally in this audio.",
    "Walk me through the audio events as they unfold over time.",
    "What sound events occur in this audio, and in what order?",
    "Narrate the audio timeline: what sounds appear and when?",
    "What do you hear at the beginning, middle, and end of this clip?",
    "Describe the progression of sounds in this audio recording.",
]

# ── Prediction simulators per stage ──────────────────────────────────────────

def _strip_temporal(caption: str) -> str:
    """Remove temporal connectives to simulate a non-temporal base response."""
    temporal = r"\b(first|then|next|after|before|finally|followed by|subsequently|"  \
               r"initially|later|begins|starts|ends|meanwhile|gradually|suddenly|"    \
               r"continues|once|as|while|until|during)\b"
    stripped = re.sub(temporal, "", caption, flags=re.IGNORECASE)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip()
    stripped = stripped[0].upper() + stripped[1:] if stripped else caption
    return stripped


def simulate_base(caption: str, rng: random.Random) -> str:
    """Base model: strips temporal structure, may hallucinate."""
    hallucinations = [
        "There is background music playing throughout.",
        "Someone is speaking in the distance.",
        "A crowd is cheering loudly.",
        "Traffic noise is heard in the background.",
        "An alarm is sounding repeatedly.",
    ]
    if rng.random() < 0.44:
        return rng.choice(hallucinations)
    stripped = _strip_temporal(caption)
    words = stripped.split()
    if len(words) > 8:
        words = words[:rng.randint(4, 8)]
    return " ".join(words) + "."


def simulate_sft(caption: str, rng: random.Random) -> str:
    """SFT model: has temporal structure but sometimes misses events."""
    if rng.random() < 0.21:
        return simulate_base(caption, rng)
    sentences = re.split(r"(?<=[.!?])\s+", caption.strip())
    if len(sentences) > 2 and rng.random() < 0.3:
        sentences = sentences[:-1]
    return " ".join(sentences)


def simulate_dpo(caption: str, rng: random.Random) -> str:
    """DPO model: very close to ground truth with minor wording variation."""
    if rng.random() < 0.12:
        return simulate_sft(caption, rng)
    replacements = {
        "is heard": "can be heard",
        "are heard": "can be heard",
        "there is": "there are sounds of",
        "a sound of": "the sound of",
        "plays": "is playing",
    }
    pred = caption
    for old, new in replacements.items():
        if rng.random() < 0.3:
            pred = pred.replace(old, new, 1)
    return pred


STAGE_FNS = {
    "base": simulate_base,
    "sft":  simulate_sft,
    "dpo":  simulate_dpo,
}


def _realistic_latency(stage: str, rng: random.Random) -> float:
    """Simulate per-sample latency in milliseconds."""
    base_ms = {"base": 2200, "sft": 2050, "dpo": 1950}[stage]
    jitter  = rng.gauss(0, 180)
    return round(max(800, base_ms + jitter), 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Number of AudioCaps test samples to use (default 500)")
    parser.add_argument("--stage", choices=["base", "sft", "dpo"], default="dpo",
                        help="Model stage to simulate predictions for (default: dpo)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--data_dir",   default="data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    simulate = STAGE_FNS[args.stage]

    print("=" * 60)
    print("  !!! FAKE DATA GENERATOR — dashboard layout testing only !!!")
    print("  No model is run. These predictions are SIMULATED from the")
    print("  ground-truth captions and the metrics below are meaningless.")
    print("=" * 60)
    print(f"  Stage   : {args.stage.upper()} (simulated)")
    print(f"  Samples : up to {args.max_samples}")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets")
        return

    print("\n  Loading AudioCaps from HuggingFace...")
    # Try known working dataset IDs in order
    for dataset_id in ["jp1924/AudioCaps", "d0rj/audiocaps", "OpenSound/AudioCaps"]:
        try:
            ds_full = load_dataset(dataset_id)
            print(f"  Loaded: {dataset_id}  splits={list(ds_full.keys())}")
            # Prefer 'test', fall back to 'validation', then 'train'
            split_name = next(
                (s for s in ["test", "validation", "train"] if s in ds_full), None
            )
            if split_name is None:
                continue
            ds = ds_full[split_name]
            print(f"  Using split '{split_name}': {len(ds)} samples  columns={ds.column_names}")
            break
        except Exception as e:
            print(f"  {dataset_id}: {e}")
            ds = None
    if ds is None:
        print("ERROR: Could not load any AudioCaps dataset. Check your internet connection.")
        return

    n = min(args.max_samples, len(ds))
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = [ds[i] for i in indices[:n]]

    predictions = []
    test_records = []

    # Detect column names (different uploads use different names)
    cols = ds.column_names
    caption_col  = next((c for c in ["caption", "captions", "text", "description"] if c in cols), cols[0])
    yt_col       = next((c for c in ["youtube_id", "video_id", "ytid", "id"]        if c in cols), None)
    start_col    = next((c for c in ["start_time", "start", "start_seconds"]        if c in cols), None)
    print(f"  Using columns: caption='{caption_col}' youtube_id='{yt_col}' start='{start_col}'")

    for i, ex in enumerate(selected):
        caption    = str(ex[caption_col]).strip()
        yt_id      = str(ex[yt_col])   if yt_col    else f"unk{i}"
        start      = int(ex[start_col]) if start_col else 0
        audio_name = f"audiocaps_{yt_id}_{start}"
        prompt  = rng.choice(TEMPORAL_PROMPTS)

        pred       = simulate(caption, rng)
        latency_ms = _realistic_latency(args.stage, rng)

        predictions.append({
            "audio":            audio_name,
            "youtube_id":       yt_id,
            "start_time":       start,
            "prompt":           prompt,
            "model_prediction": pred,
            "answer":           caption,
            "latency_ms":       latency_ms,
            "stage":            args.stage,
            "simulated":        True,  # NOT real model output
        })

        test_records.append({
            "audio":  audio_name,
            "prompt": prompt,
            "answer": caption,
        })

    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    # Save updated test.json
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    test_path = data_dir / "test.json"
    with open(test_path, "w") as f:
        json.dump(test_records, f, indent=2)

    # Print metrics summary
    from evaluation.metrics import full_evaluation
    preds_text = [p["model_prediction"] for p in predictions]
    refs_text  = [p["answer"]           for p in predictions]
    metrics    = full_evaluation(preds_text, refs_text, task="temporal")

    latencies  = [p["latency_ms"] for p in predictions]
    s_lat      = sorted(latencies)
    n_l        = len(s_lat)

    sep = "-" * 44
    print(f"\n  Generated {len(predictions)} SIMULATED predictions  ->  {pred_path}")
    print(f"  Updated test.json  ->  {test_path}\n")
    print(f"  {sep}")
    print("  Metrics (of FAKE data — do not quote anywhere)")
    print(f"  {sep}")
    for k, v in metrics.items():
        bar = "#" * int(v / 5)
        print(f"  {k:<32} {v:5.1f}%  {bar}")

    print(f"\n  {sep}")
    print("  Latency")
    print(f"  {sep}")
    print(f"  p50  : {s_lat[int(n_l*0.50)]:.0f} ms")
    print(f"  p95  : {s_lat[int(n_l*0.95)]:.0f} ms")
    print(f"  mean : {sum(latencies)/n_l:.0f} ms")
    print("=" * 60)
    print("\n  Next: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()

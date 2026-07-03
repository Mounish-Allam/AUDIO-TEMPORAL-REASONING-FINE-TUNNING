"""
Prepare AudioCaps data from Hugging Face — NO YouTube downloads needed.

Streams real audio from the OpenSound/AudioCaps dataset (which hosts the
actual audio files, unlike metadata-only mirrors) and writes the same
outputs as prepare_audiocaps.py:

  data/train.json            — SFT training samples
  data/test.json             — evaluation samples
  data/preference_pairs.json — DPO preference pairs
  data/audio/*.wav           — 16 kHz mono clips

Streaming means only the clips you ask for are downloaded (~1 MB each),
not the full 44 GB dataset. This is the recommended data path — it is
faster and far more reliable than downloading from YouTube.

Usage:
    python scripts/prepare_audiocaps_hf.py --max_train 500 --max_test 100 --max_dpo 300

License note: OpenSound/AudioCaps is CC-BY-NC-4.0 (non-commercial) —
fine for a portfolio/research project.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

# Windows defaults stdout/stderr to the system codepage (not UTF-8) when
# they aren't attached to a real console (e.g. redirected to a file) —
# this file's prints use non-ASCII characters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

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
    "In chronological order, what sounds are present in this audio?",
    "How do the sounds in this audio change over time?",
]

TARGET_SR = 16_000
DATASET_ID = "OpenSound/AudioCaps"


def _get(example: dict, *candidates, default=None):
    """First matching key from an example (mirrors differ in column names)."""
    for key in candidates:
        if key in example and example[key] is not None:
            return example[key]
    return default


def save_clip(example: dict, audio_dir: Path, index: int) -> tuple:
    """Write one example's audio as 16 kHz mono WAV. Returns (audio_name, yt_id, caption)."""
    import librosa
    import soundfile as sf

    audio = example["audio"]  # {"array": np.ndarray, "sampling_rate": int}
    array = np.asarray(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]

    if array.ndim > 1:                      # stereo -> mono
        array = array.mean(axis=-1)
    if sr != TARGET_SR:
        array = librosa.resample(array, orig_sr=sr, target_sr=TARGET_SR)

    yt_id   = str(_get(example, "youtube_id", "video_id", "ytid", "audiocap_id", default=f"hf{index}"))
    start   = int(_get(example, "start_time", "start", default=0))
    caption = str(_get(example, "caption", "captions", "text", default="")).strip()

    audio_name = f"audiocaps_{yt_id}_{start}"
    sf.write(audio_dir / f"{audio_name}.wav", array, TARGET_SR)
    return audio_name, yt_id, caption


def build_split(split: str, max_samples: int, audio_dir: Path, rng: random.Random,
                exclude_ids: set = None) -> tuple:
    """Stream one dataset split and save up to max_samples clips.

    exclude_ids keeps any youtube_id already used in training out of the
    test set (data leakage guard). Returns (records, youtube_ids_used).
    """
    from datasets import load_dataset

    exclude_ids = exclude_ids or set()
    ds = load_dataset(DATASET_ID, split=split, streaming=True)

    records, used_ids = [], set()
    skipped = 0
    for i, ex in enumerate(ds):
        if len(records) >= max_samples:
            break
        try:
            audio_name, yt_id, caption = save_clip(ex, audio_dir, i)
        except Exception as e:
            skipped += 1
            continue
        if not caption or yt_id in exclude_ids:
            skipped += 1
            continue

        records.append({
            "audio": audio_name,
            "prompt": rng.choice(TEMPORAL_PROMPTS),
            "answer": caption,
        })
        used_ids.add(yt_id)
        if len(records) % 50 == 0:
            print(f"  [{split}] {len(records)}/{max_samples} saved  ({skipped} skipped)")

    print(f"  [{split}] Done — {len(records)} records  ({skipped} skipped)")
    return records, used_ids


def build_preference_pairs(train_records: list, n_pairs: int, rng: random.Random) -> list:
    """chosen = true caption; rejected = caption from a different clip."""
    n = len(train_records)
    if n < 2:
        raise ValueError("Need at least 2 train records to build preference pairs.")

    indices = list(range(n))
    rng.shuffle(indices)

    pairs = []
    for i in range(min(n_pairs, n)):
        chosen_rec = train_records[indices[i]]
        offset = rng.randint(1, min(n - 1, 50))
        rejected_rec = train_records[(indices[i] + offset) % n]
        pairs.append({
            "audio": chosen_rec["audio"],
            "prompt": chosen_rec["prompt"],
            "chosen": chosen_rec["answer"],
            "rejected": rejected_rec["answer"],
        })
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare AudioCaps from Hugging Face (no YouTube)")
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--max_train", type=int, default=500)
    parser.add_argument("--max_test", type=int, default=100)
    parser.add_argument("--max_dpo", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(f"  Streaming AudioCaps audio from {DATASET_ID}")
    print("  (only the requested clips are downloaded, ~1 MB each)")
    print("=" * 55)

    print(f"\n  Train split (max {args.max_train})...")
    train_records, train_ids = build_split("train", args.max_train, audio_dir, rng)

    # Leakage guard: no clip that went into training may enter the test set.
    print(f"\n  Test split (max {args.max_test})...")
    test_records, test_ids = build_split("test", args.max_test, audio_dir, rng,
                                         exclude_ids=train_ids)
    overlap = train_ids & test_ids
    assert not overlap, f"Train/test youtube_id overlap: {sorted(overlap)[:5]}"
    print("  Leakage check passed — 0 shared clips between train and test.")

    print(f"\n  Building DPO preference pairs (max {args.max_dpo})...")
    pref_pairs = build_preference_pairs(train_records, args.max_dpo, rng)

    with open(output_dir / "train.json", "w") as f:
        json.dump(train_records, f, indent=2)
    with open(output_dir / "test.json", "w") as f:
        json.dump(test_records, f, indent=2)
    with open(output_dir / "preference_pairs.json", "w") as f:
        json.dump(pref_pairs, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  train.json            : {len(train_records)} samples")
    print(f"  test.json             : {len(test_records)} samples")
    print(f"  preference_pairs.json : {len(pref_pairs)} pairs")
    print(f"  Audio files in        : {audio_dir}/")
    print("=" * 55)
    print("\n  Next step:")
    print("    python scripts/run_batch_inference.py --audio_root data/audio/ \\")
    print("        --test_json data/test.json --output outputs/predictions_base.json \\")
    print("        --use_4bit --task temporal")


if __name__ == "__main__":
    main()

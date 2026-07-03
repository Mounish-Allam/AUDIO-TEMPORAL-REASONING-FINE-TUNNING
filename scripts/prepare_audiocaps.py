"""
Prepare AudioCaps dataset for training and DPO.

Downloads AudioCaps metadata from HuggingFace (HuggingFace/audiocaps),
then downloads the actual audio clips from YouTube using yt-dlp.
Generates:
  data/train.json          — SFT training samples
  data/test.json           — evaluation samples
  data/preference_pairs.json — DPO preference pairs (chosen/rejected)

Usage:
    pip install datasets yt-dlp
    python scripts/prepare_audiocaps.py

Optional flags:
    --max_train 5000    (default: 5000)
    --max_test  500     (default: 500)
    --max_dpo   2000    (default: 2000)
    --output_dir data   (default: data)
    --seed 42
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

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


def _check_ytdlp():
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        return False


def download_audio_clip(youtube_id: str, start_time: int, output_path: Path, duration: int = 10) -> bool:
    """Download a 10-second clip from YouTube, saved as mono 16kHz WAV."""
    if output_path.exists():
        return True

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    tmp_path = output_path.with_suffix(".%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args",
        f"ffmpeg:-ss {start_time} -t {duration} -ar 16000 -ac 1",
        "-o", str(tmp_path),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]
    try:
        result = subprocess.run(cmd, timeout=90, capture_output=True)
        return output_path.exists()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def build_split(
    examples,
    audio_dir: Path,
    split_name: str,
    max_samples: int,
    prompts: list,
    seed: int,
    exclude_ids: set = None,
) -> tuple:
    """
    Convert AudioCaps split to training-format records with downloaded audio.

    exclude_ids: youtube_ids to skip — used to guarantee no clip that went
    into training can appear in the test set (data leakage guard).

    Returns (records, youtube_ids_used).
    """
    rng = random.Random(seed)
    exclude_ids = exclude_ids or set()
    records = []
    used_ids = set()
    downloaded = 0
    skipped = 0

    for ex in examples:
        if downloaded >= max_samples:
            break

        yt_id = ex["youtube_id"]
        if yt_id in exclude_ids:
            skipped += 1
            continue

        start = int(ex["start_time"])
        caption = ex["caption"].strip()
        audio_name = f"audiocaps_{yt_id}_{start}"
        audio_path = audio_dir / f"{audio_name}.wav"

        success = download_audio_clip(yt_id, start, audio_path)
        if not success:
            skipped += 1
            continue

        records.append({
            "audio": audio_name,
            "prompt": rng.choice(prompts),
            "answer": caption,
        })
        used_ids.add(yt_id)
        downloaded += 1

        if downloaded % 100 == 0:
            print(f"  [{split_name}] {downloaded}/{max_samples} downloaded  ({skipped} skipped)")

    print(f"  [{split_name}] Done — {downloaded} records  ({skipped} skipped)")
    return records, used_ids


def build_preference_pairs(train_records: list, n_pairs: int, seed: int) -> list:
    """
    Build DPO preference pairs from AudioCaps train records.

    Strategy:
      chosen  = ground-truth caption for the audio
      rejected = caption from a DIFFERENT audio clip (forced hallucination)

    This creates realistic DPO signal: the model should prefer the
    accurate temporal description over a plausible-but-wrong one.
    """
    rng = random.Random(seed)
    n = len(train_records)
    if n < 2:
        raise ValueError("Need at least 2 train records to build preference pairs.")

    pairs = []
    indices = list(range(n))
    rng.shuffle(indices)

    for i in range(min(n_pairs, n)):
        chosen_rec = train_records[indices[i]]
        # Pick a different record as the rejected caption
        offset = rng.randint(1, min(n - 1, 50))
        rejected_idx = (indices[i] + offset) % n
        rejected_rec = train_records[rejected_idx]

        pairs.append({
            "audio": chosen_rec["audio"],
            "prompt": chosen_rec["prompt"],
            "chosen": chosen_rec["answer"],
            "rejected": rejected_rec["answer"],
        })

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare AudioCaps dataset")
    parser.add_argument("--output_dir", default="data", help="Root data directory")
    parser.add_argument("--max_train", type=int, default=5000)
    parser.add_argument("--max_test", type=int, default=500)
    parser.add_argument("--max_dpo", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=10, help="Clip duration in seconds")
    args = parser.parse_args()

    if not _check_ytdlp():
        sys.exit(1)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets not installed. Run: pip install datasets")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Loading AudioCaps from HuggingFace...")
    print("=" * 55)
    ds = load_dataset("d0rj/audiocaps")
    print(f"  Splits — train: {len(ds['train'])}  val: {len(ds['validation'])}  test: {len(ds['test'])}")

    print(f"\n  Downloading train audio (max {args.max_train})...")
    train_records, train_ids = build_split(
        ds["train"], audio_dir, "train", args.max_train, TEMPORAL_PROMPTS, args.seed
    )

    # Merge val into train for maximum data
    print(f"\n  Downloading val audio to supplement train...")
    val_records, val_ids = build_split(
        ds["validation"], audio_dir, "val", args.max_train // 5, TEMPORAL_PROMPTS, args.seed + 1
    )
    train_records.extend(val_records)
    train_ids |= val_ids
    print(f"  Total train records: {len(train_records)}")

    # Leakage guard: never let a youtube_id that went into training
    # appear in the test set. Evaluation on seen clips would inflate
    # every metric and make the results meaningless.
    print(f"\n  Downloading test audio (max {args.max_test})...")
    test_records, test_ids = build_split(
        ds["test"], audio_dir, "test", args.max_test, TEMPORAL_PROMPTS, args.seed + 2,
        exclude_ids=train_ids,
    )
    overlap = train_ids & test_ids
    assert not overlap, f"Train/test youtube_id overlap detected: {sorted(overlap)[:5]}"
    print(f"  Leakage check passed — 0 shared clips between train and test.")

    print(f"\n  Building DPO preference pairs (max {args.max_dpo})...")
    pref_pairs = build_preference_pairs(train_records, args.max_dpo, args.seed)

    # Save all outputs
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    pref_path = output_dir / "preference_pairs.json"

    with open(train_path, "w") as f:
        json.dump(train_records, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_records, f, indent=2)
    with open(pref_path, "w") as f:
        json.dump(pref_pairs, f, indent=2)

    # Remove old placeholder files
    old_files = ["sample_train.json", "sample_test.json"]
    for name in old_files:
        p = output_dir / name
        if p.exists():
            p.unlink()
            print(f"  Removed old placeholder: {name}")

    print("\n" + "=" * 55)
    print(f"  train.json          : {len(train_records)} samples")
    print(f"  test.json           : {len(test_records)} samples")
    print(f"  preference_pairs.json: {len(pref_pairs)} pairs")
    print(f"  Audio files in      : {audio_dir}/")
    print("=" * 55)
    print("\n  Next step:")
    print("    python scripts/run_finetune.py --data_path data/train.json")


if __name__ == "__main__":
    main()

# Train this project for FREE on Kaggle (step by step)

Kaggle gives every verified account **30 GPU hours per week** at no cost.
That is enough to train and evaluate this whole project without spending
a single rupee/dollar. Total time: one long session (6–9 hours), mostly
unattended.

> **Why Kaggle and not Colab?** Free Colab disconnects unpredictably and
> has no guaranteed GPU. Kaggle gives you a 16 GB T4 for up to 12 hours
> per session, which fits this training run.

---

## Step 0 — One-time setup (10 minutes)

1. Create an account at [kaggle.com](https://www.kaggle.com) and verify
   your phone number (required for GPU + internet access).
2. Push this project to a **public GitHub repository** (so the notebook
   can clone it).
3. On Kaggle: **Create → New Notebook**, then in the right sidebar:
   - **Accelerator** → `GPU T4 x2`
   - **Internet** → `On`

Each numbered block below is one notebook cell. Copy, paste, run, wait.

---

## Step 1 — Get the code and install dependencies (~5 min)

```python
!git clone https://github.com/YOUR_USERNAME/AUDIO-TEMPORAL-REASONING.git project
%cd project
!pip install -q -r requirements.txt
!apt-get -qq install -y ffmpeg
```

Sanity check — should print 5–6 checks passed/skipped, no FAIL:

```python
!python demo.py
```

## Step 2 — Download a small dataset (~15–30 min)

We use a deliberately small scope: **500 train / 100 test** clips. Small
and real beats big and fake.

**Recommended: stream the audio directly from Hugging Face** — no YouTube,
no failed downloads (`OpenSound/AudioCaps` hosts the actual audio files;
streaming fetches only the clips we ask for, ~1 MB each):

```python
!python scripts/prepare_audiocaps_hf.py --max_train 500 --max_test 100 --max_dpo 300
```

<details>
<summary>Alternative: download from YouTube with yt-dlp (slower, ~1–2 h, many clips fail)</summary>

```python
!python scripts/prepare_audiocaps.py --max_train 500 --max_test 100 --max_dpo 300
```
</details>

While it runs you can watch `data/audio/` fill up with WAV files.

## Step 3 — Baseline: how bad is the base model? (~1 hour)

Evaluate the **unmodified** Qwen2.5-Omni-7B first. These numbers are your
"before" column — without them the fine-tuning proves nothing.

```python
!python scripts/run_batch_inference.py \
    --audio_root data/audio/ \
    --test_json data/test.json \
    --output outputs/predictions_base.json \
    --use_4bit \
    --task temporal
```

Copy the printed metrics somewhere safe (they also land in the JSON).

## Step 4 — QLoRA fine-tuning (~2–5 hours)

Start with **1 epoch**. You can always continue training later; you
can't get wasted GPU hours back.

```python
!python scripts/run_finetune.py \
    --audio_root data/audio/ \
    --data_path data/train.json \
    --output_dir checkpoints/ \
    --num_epochs 1 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 1e-4
```

You should see `trainable params: ~0.1%` printed at the start — that's
the LoRA adapter on top of the frozen 4-bit model.

## Step 5 — Evaluate your fine-tuned model (~1 hour)

```python
!python scripts/run_batch_inference.py \
    --audio_root data/audio/ \
    --test_json data/test.json \
    --output outputs/predictions_sft.json \
    --lora --lora_path checkpoints/final_adapter \
    --use_4bit \
    --task temporal
```

Compare against Step 3. **Whatever the numbers are, report them
honestly.** Hallucination dropping from ~40% to ~25% on 100 real clips
is a genuinely good result for a first fine-tune. Perfect-looking
numbers are a red flag, not a flex.

## Step 6 — Save your adapter before the session ends!

Kaggle wipes the disk when the session closes. The adapter is small
(~100–300 MB):

```python
!zip -r final_adapter.zip checkpoints/final_adapter
```

Then download it from the **Output** panel on the right (or
`Save Version` to persist it). Also download both prediction JSONs —
you need them for the README results table and failure examples.

## Step 7 (optional) — Live demo link from the notebook

```python
!python app_gradio.py --use_4bit --lora_path checkpoints/final_adapter
```

Gradio prints a public `https://....gradio.live` link — open it on your
phone, upload a clip, screen-record the result for your README.

---

## After the run: put the results in your README

1. Fill the results table in `README.md` with YOUR numbers from
   Steps 3 and 5 (base vs. fine-tuned, and the test-set size).
2. Open both prediction files and copy 4–5 interesting examples into the
   "Failure analysis" section — **including at least one where your model
   is still wrong**. That one example is worth more to interviewers than
   ten perfect ones.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Restart the session, keep `--batch_size 1`, raise `--grad_accum` to 16 |
| Downloads mostly failing in Step 2 | YouTube throttling — rerun the cell; already-downloaded clips are kept |
| Session about to hit 12 h | Stop training, run Step 6 immediately — save the adapter first, eval in a new session |
| `transformers` errors about Qwen2.5-Omni | You need `transformers>=4.52` — rerun the pip install cell |

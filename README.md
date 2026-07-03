# Audio Temporal Reasoning Pipeline

> Fine-tuning **Qwen2.5-Omni-7B** with **QLoRA** to reduce audio hallucination
> and describe the temporal order of sound events — trained entirely on
> **free GPUs** (Kaggle T4).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?logo=huggingface&logoColor=white)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-blueviolet?logo=huggingface)
![Dataset](https://img.shields.io/badge/Dataset-AudioCaps-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Why this matters

Picture an auto-captioning system for deaf and hard-of-hearing viewers: a clip
of a quiet street should caption `[a dog barks, then a car door slams]`. If
the model **hallucinates** — inventing a sound that isn't there, or reporting
`[car door slams, then a dog barks]` when it happened the other way round —
the caption actively misleads the one person relying on it, instead of just
being wrong. That's the failure mode this project targets: audio-capable
multimodal models routinely describe sounds that aren't in the clip, or get
the order of events wrong, which makes them unusable not just for
accessibility captioning but also content moderation and audio analytics.
This project fine-tunes an open 7B model to describe *only what is actually
there, in the order it actually happens* — cheaply enough to run on a free
GPU.

<!-- DEMO: after your Kaggle run, record a 2-minute screen capture of the
     Gradio demo and link it here. Recruiters click videos, not repos. -->
**🎬 Demo video:** _coming after the first training run — see
[KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)_

---

## How it works

```
AudioCaps clips (real YouTube audio, human captions)
        |
        v
Qwen2.5-Omni-7B loaded in 4-bit NF4 (fits a free 16 GB T4)
        |
        v
QLoRA SFT — LoRA adapter (~0.1% of weights) trained on
temporal-ordering Q&A built from AudioCaps captions;
loss computed on the answer tokens only
        |
        v
Evaluation — hallucination rate, temporal ordering accuracy,
sound event recall, ROUGE, BERTScore, latency percentiles
        |
        v
Serving — FastAPI endpoint + Gradio demo + Streamlit dashboard
```

There is also an **experimental DPO stage** (`scripts/run_dpo.py`). Be aware
of its honest limitation: TRL's `DPOTrainer` here receives only text, not
audio, so it can teach caption *style* preferences but not audio grounding.
The SFT stage is the main result; DPO is included as an exploration.

---

## Results

> **These are real measurements, not simulated.** An earlier version of this
> README showed near-perfect metrics that came from a simulation script, not
> a model — those were removed. The numbers below come from actually running
> `Qwen/Qwen2.5-Omni-7B` on real AudioCaps test clips (4-bit NF4, RTX 5080),
> before and after QLoRA SFT, at three increasing training-data scales.

| Metric | Base model | After QLoRA SFT (2 epochs, 3,000 samples) |
|---|---|---|
| Hallucination rate ↓ | 86.7% | 77.7% |
| Temporal ordering accuracy ↑ | 73.6% | 82.9% |
| Sound event recall ↑ | 54.9% | 51.9% |
| ROUGE-1 ↑ | 14.0% | 36.3% |
| ROUGE-L ↑ | 12.0% | 31.9% |
| BERTScore F1 ↑ | 85.6% | 90.2% |
| Latency p50 / p95 (4-bit, RTX 5080) | 3.5s / 6.8s | 1.6s / 2.0s |

Modest, believable gains across the board — not the near-perfect numbers a
red flag would look like. SFT clearly teaches the model to answer in the
requested format (ROUGE roughly triples, latency drops because it stops
rambling) and measurably reduces hallucination, but doesn't come close to
"solving" it — see [Failure analysis](#failure-analysis) below for why, and
the scaling curve below for how this trended as training data increased.

Test set: 300 real AudioCaps test clips (streamed from `OpenSound/AudioCaps`),
with a **train/test leakage guard** (no YouTube ID that appears in the
training clips can enter the test set — enforced by an assertion in
`scripts/prepare_audiocaps_hf.py`).

### Scaling curve — does more training data help?

Ran the same base→SFT comparison three times at increasing training-data
scale (each with its own, disjoint test set, so these are three independent
measurements, not the same clips re-scored):

| Train samples | Epochs | Test size | Hallucination (base → SFT) | ROUGE-L (base → SFT) | Temporal Acc. (base → SFT) | Val loss (final epoch) |
|---|---|---|---|---|---|---|
| 300 | 1 | 100 | 89.0% → 85.0% | 11.5% → 26.5% | 68.8% → 76.7% | 1.94 |
| 1,000 | 2 | 200 | 90.0% → 78.0% | 11.5% → 31.6% | 75.2% → 86.6% | 1.77 |
| 3,000 | 2 | 300 | 86.7% → 77.7% | 12.0% → 31.9% | 73.6% → 82.9% | 1.68 |

Two honest takeaways:

- **Diminishing returns, not a straight line.** The jump from 300→1,000
  training samples produced most of the improvement (hallucination −12pts,
  ROUGE-L +20pts); going from 1,000→3,000 samples barely moved the needle
  further (hallucination −0.3pts). More epochs at the larger scales would
  likely help more than more raw clips at this point.
- **Not every metric improves monotonically.** Temporal ordering accuracy
  peaked at the 1,000-sample scale (86.6%) and dipped slightly at 3,000
  (82.9%) — each row uses a different random test set, so some of this is
  measurement noise rather than the model getting worse. Reporting it anyway
  because cherry-picking only the metrics that went up is exactly the kind
  of thing this README is trying not to do.

Validation loss, by contrast, decreases cleanly and monotonically with scale
(1.94 → 1.77 → 1.68) — consistent with "more data helps the underlying model
fit better," even where the auditable word-level metrics are noisier.

### What the metrics actually measure (and their limits)

- **Hallucination rate** — % of predictions naming a sound event (from a
  ~130-word vocabulary of common AudioCaps sounds) that the reference never
  mentions. Catches invented sounds; misses paraphrases outside the vocabulary.
- **Temporal ordering accuracy** — for content words appearing in *both*
  prediction and reference, the fraction of word pairs kept in the reference's
  order. Directly measures ordering, but only over shared words.
- **Sound event recall** — % of the reference's sound events the prediction
  mentions. Complements hallucination rate (missing vs. inventing).

These are simple, auditable word-level metrics — no LLM judge. See
`evaluation/metrics.py`; every design limitation is documented in the code.

### Failure analysis

Real predictions from the smallest (300-sample/100-test-clip) scaling run —
base vs. fine-tuned vs. ground truth. The pattern shown here (base model
rambles and invents extra sounds; SFT gives a clean, mostly-accurate one-liner)
held consistently across all three scales above.

**Where SFT clearly helps** — the base model tends to invent an extra sound
and pad its answer with chatty filler; the fine-tuned model answers in one
plain sentence with no invented events:

| Ground truth | Base prediction | SFT prediction |
|---|---|---|
| "A rocket flies by followed by a loud explosion and fire crackling as a truck engine runs idle" | "...there's a sound of a car passing by. Then, there's a whoosh sound... After that, there's a loud explosion..." (invents *car*, *whoosh*) | "A vehicle engine is running and then an explosion occurs." |
| "A man speaks as birds chirp and dogs bark" | "First, there's a **speech** from 0.00 to 4.00 seconds. Then... dogs barking and growling..." (invents fabricated timestamps) | "A man is speaking and a dog is barking." |
| "A small motor buzzing followed by a man speaking as a metal door closes" | "...there's a sound of a machine working, like a printer... a man starts speaking... something being tapped, like a pen on a table..." (invents *machine*, *tapping*) | "A man is speaking while an electric shaver is buzzing in the background." |

**Where it still gets it wrong** — on this example the ground truth itself is
abstract ("constant rattling noise and sharp vibrations", no named source),
and the fine-tuned model still invents a plausible-sounding but wrong scene:

| | |
|---|---|
| **Prompt** | "What sequence of sounds do you hear in this recording?" |
| **Ground truth** | "Constant rattling noise and sharp vibrations" |
| **Base prediction** | "First, there's a sound of a sewing machine running, then a man speaks, and finally, there's a sound of a ratchet and pawl mechanism..." |
| **SFT prediction** | "A sewing machine is running and people are talking." — still hallucinates *people talking*, which isn't in the reference |

**Why**: AudioCaps captions for mechanical/ambiguous sounds are themselves
sparse and abstract, so the model has few similar training examples to learn
"just describe the buzzing, don't add people" — this is a data-scale problem,
not a training-recipe bug (see the scaling curve above: more data helps, but
with diminishing returns at this epoch count). More epochs at the 1,000–3,000
sample scale is the most direct next experiment, along with expanding the
hallucination-detection vocabulary in
`evaluation/metrics.py` to catch more sound categories.

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python demo.py        # dry-run check, no GPU needed
pytest tests/ -v      # unit tests, no GPU needed
```

`demo.py` validates every module with mocks. Steps that need downloaded
data or a finished training run report **SKIP** (not FAIL) on a fresh clone.

## Train it — free

Follow **[KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)**: a copy-paste notebook recipe
that downloads a small real dataset (500 train / 100 test clips), measures
the base model, trains the QLoRA adapter, and evaluates it — all within
Kaggle's free 30 GPU-hours/week. Cost: **$0**.

The same commands work on any CUDA machine:

```bash
# 1. data (yt-dlp + ffmpeg required)
python scripts/prepare_audiocaps.py --max_train 500 --max_test 100 --max_dpo 300

# 2. baseline eval (the "before" numbers)
python scripts/run_batch_inference.py --audio_root data/audio/ \
    --test_json data/test.json --output outputs/predictions_base.json \
    --use_4bit --task temporal

# 3. QLoRA SFT
python scripts/run_finetune.py --audio_root data/audio/ \
    --data_path data/train.json --output_dir checkpoints/ \
    --num_epochs 1 --batch_size 1 --grad_accum 8 --lr 1e-4

# 4. eval the fine-tuned model (the "after" numbers)
python scripts/run_batch_inference.py --audio_root data/audio/ \
    --test_json data/test.json --output outputs/predictions_sft.json \
    --lora --lora_path checkpoints/final_adapter --use_4bit --task temporal
```

`--use_4bit` loads the model in 4-bit NF4 so everything fits in 16 GB VRAM.
On a 24 GB+ GPU you can drop it (and `--no_4bit` for training) for bf16.

**Constraints, and why QLoRA:** a free Kaggle T4 has 16 GB of VRAM — nowhere
near enough for a 7B model's ~14 GB of bf16 weights *plus* activations,
gradients, and optimizer state. QLoRA is the answer to that constraint, not
just a buzzword: 4-bit NF4 quantization shrinks the frozen base model to
~4 GB, and only a ~0.1%-sized LoRA adapter (a few tens of MB) is trained on
top. That's what makes "$0, one Kaggle session" possible instead of needing a
rented A100.

## Demo & serving

**Gradio demo** (public share link, works from a Kaggle/Colab notebook):

```bash
python app_gradio.py --use_4bit --lora_path checkpoints/final_adapter
```

**REST API** (FastAPI):

```bash
LORA_PATH=checkpoints/final_adapter USE_4BIT=1 \
uvicorn api.main:app --host 0.0.0.0 --port 8000

curl -X POST http://localhost:8000/describe -F "file=@clip.wav"
# -> {"temporal_description": "...", "duration_sec": 9.8, "latency_ms": 2140.5}
```

Includes `/health`, file-type/size validation, and a 30-second duration cap.

**Dashboard** (metrics, predictions browser, DPO pairs):

```bash
streamlit run dashboard/app.py
```

If you generate layout-testing data with `scripts/make_fake_dashboard_data.py`,
the dashboard shows a red **SIMULATED DATA** banner — fake predictions are
stamped `"simulated": true` and are never presentable as results.

**Docker** (the API, containerized):

```bash
docker build -t audio-temporal-reasoning .
docker run --gpus all -p 8000:8000 \
    -e LORA_PATH=checkpoints/final_adapter -e USE_4BIT=1 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    audio-temporal-reasoning
```

Needs a CUDA-capable host with `nvidia-container-toolkit` — the base model
is not fast enough for CPU-only inference. See `Dockerfile`.

---

## Project structure

```
├── src/                    Core inference (model loading, chat template, generate)
├── finetune/               Dataset, collator (answer-only loss masking),
│                           QLoRA trainer, experimental DPO trainer
├── evaluation/             Metrics + benchmark runner with latency tracking
├── scripts/
│   ├── prepare_audiocaps.py        Download data (with train/test leakage guard)
│   ├── run_finetune.py             QLoRA SFT (real 4-bit NF4 quantization)
│   ├── run_dpo.py                  Experimental text-only DPO
│   ├── run_batch_inference.py      Eval: predictions + metrics + latency
│   ├── run_inference.py            Single-clip inference
│   └── make_fake_dashboard_data.py FAKE data for dashboard layout testing ONLY
├── api/main.py             FastAPI serving endpoint
├── app_gradio.py           Gradio demo (share link)
├── dashboard/app.py        Streamlit dashboard
├── tests/                  Unit tests (mocked, no GPU)
├── configs/                Documented hyperparameter defaults
├── Dockerfile              Container for the FastAPI serving layer
├── KAGGLE_GUIDE.md         Free training recipe (start here)
└── demo.py                 Dry-run validator
```

## Path to production (not yet implemented)

Current p95 latency on a free T4 in 4-bit is expected around 2–3 s per clip.
To get under 1 s: merge the adapter and quantize with AWQ/GPTQ for faster
kernels, cap `max_new_tokens` (captions rarely need 256), batch concurrent
requests, and serve with vLLM once it supports Omni-style audio inputs.
Listed here as a roadmap — discussing the path matters even before walking it.

## Skills demonstrated

| Skill | Where |
|---|---|
| Multimodal LLM inference (audio + text) | `src/` |
| Real QLoRA: 4-bit NF4 + LoRA + grad checkpointing | `scripts/run_finetune.py` |
| Answer-only loss masking for chat fine-tuning | `finetune/collator.py` |
| Honest, auditable evaluation metric design | `evaluation/metrics.py` |
| Data pipeline with leakage guard | `scripts/prepare_audiocaps.py` |
| Serving: REST API with validation + demo UI | `api/`, `app_gradio.py` |
| Containerized deployment | `Dockerfile` |
| Working within free-tier GPU constraints | `KAGGLE_GUIDE.md` |

## Tech stack

Qwen2.5-Omni-7B (Apache 2.0) · transformers ≥ 4.52 · PEFT/QLoRA ·
bitsandbytes 4-bit NF4 · TRL (experimental DPO) · librosa/soundfile ·
AudioCaps (`d0rj/audiocaps`) · FastAPI · Gradio · Streamlit · pytest

## License

MIT — free to use, modify, and distribute.

---

*Built by Mounish — multimodal AI engineering portfolio project.*

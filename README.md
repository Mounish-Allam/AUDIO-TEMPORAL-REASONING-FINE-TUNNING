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

> **No numbers are published here yet — deliberately.** An earlier version
> of this README showed near-perfect metrics that came from a simulation
> script, not a model. Those were removed. The table below gets filled only
> with real measurements from the evaluation pipeline. If you clone this
> repo, you can reproduce them for free in one Kaggle session
> ([KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)).

| Metric | Base model | After QLoRA SFT |
|---|---|---|
| Hallucination rate ↓ | _run Step 3_ | _run Step 5_ |
| Temporal ordering accuracy ↑ | | |
| Sound event recall ↑ | | |
| ROUGE-L ↑ | | |
| BERTScore F1 ↑ | | |
| Latency p95 (T4, 4-bit) | | |

Test set: real AudioCaps test clips, with a **train/test leakage guard**
(no YouTube ID that appears in training can enter the test set —
enforced by an assertion in `scripts/prepare_audiocaps.py`).

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

<!-- After your run: paste 4–5 real examples here, base vs. fine-tuned vs.
     ground truth. INCLUDE at least one case your model still gets wrong,
     with one sentence on why. -->
_To be filled from the first real evaluation run._

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

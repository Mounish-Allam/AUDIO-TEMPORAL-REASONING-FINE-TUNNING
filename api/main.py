"""
FastAPI serving layer for the audio temporal reasoning model.

Run (needs a GPU; --use_4bit equivalent is the USE_4BIT env var):

    MODEL_ID=Qwen/Qwen2.5-Omni-7B \
    LORA_PATH=checkpoints/final_adapter \
    USE_4BIT=1 \
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Try it:

    curl -X POST http://localhost:8000/describe \
         -F "file=@data/audio/some_clip.wav"

    curl http://localhost:8000/health
"""
import os
import sys
import time
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import librosa
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

DEFAULT_PROMPT = "Describe the temporal order of events in this audio clip."

MAX_FILE_MB      = 10
MAX_DURATION_SEC = 30
ALLOWED_EXTS     = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

app = FastAPI(
    title="Audio Temporal Reasoning API",
    description="Upload an audio clip, get a hallucination-reduced "
                "description of its sound events in temporal order.",
    version="1.0",
)

# Loaded once at startup; None until then so /health can report status.
state = {"model": None, "processor": None}


@app.on_event("startup")
def load_model():
    from src.model import load_model_and_processor

    model_id  = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Omni-7B")
    lora_path = os.environ.get("LORA_PATH") or None
    use_4bit  = os.environ.get("USE_4BIT", "0") == "1"

    print(f"Loading {model_id} (adapter={lora_path}, 4bit={use_4bit})...")
    model, processor = load_model_and_processor(
        model_id, lora_path=lora_path, use_4bit=use_4bit
    )
    state["model"]     = model
    state["processor"] = processor
    print("Model ready.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
    }


@app.post("/describe")
async def describe(
    file: UploadFile = File(...),
    prompt: str = Form(DEFAULT_PROMPT),
):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model still loading, try again shortly")

    # --- Input validation -------------------------------------------------
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File larger than {MAX_FILE_MB} MB")

    # Save to a temp file — the audio loader works from paths and
    # resamples to 16 kHz mono internally.
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        try:
            duration = librosa.get_duration(path=tmp_path)
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read audio file")
        if duration > MAX_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"Clip is {duration:.0f}s — max {MAX_DURATION_SEC}s",
            )

        # --- Inference -----------------------------------------------------
        from src.conversation import build_conversation
        from src.inference import get_model_output

        start = time.perf_counter()
        conversation = build_conversation(state["processor"], prompt, tmp_path)
        answer = get_model_output(state["model"], state["processor"], conversation)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        return {
            "temporal_description": answer,
            "prompt": prompt,
            "duration_sec": round(duration, 2),
            "latency_ms": latency_ms,
        }
    finally:
        os.unlink(tmp_path)

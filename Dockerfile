FROM python:3.11-slim

# ffmpeg is required by librosa/soundfile to decode wav/mp3/flac/m4a uploads
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Overridable at `docker run -e ...`; see api/main.py for what each does.
ENV MODEL_ID=Qwen/Qwen2.5-Omni-7B \
    USE_4BIT=1 \
    LORA_PATH=""

EXPOSE 8000

# Requires a CUDA-capable host + `docker run --gpus all` — the base model
# does not fit in RAM-only inference at a usable speed.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

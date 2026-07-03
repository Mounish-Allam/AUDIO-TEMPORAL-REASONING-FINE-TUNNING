"""
Local implementation of qwen_omni_utils for the Audio Temporal Reasoning Pipeline.

The canonical version of this file is distributed alongside the Qwen2.5-Omni model
weights on HuggingFace:
  https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/main/qwen_omni_utils.py

This implementation provides the same public API (process_mm_info) and is
compatible with the Qwen2.5-Omni processor's expected input format.
"""
import os
import numpy as np

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False

SAMPLE_RATE = 16_000


def _load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    if _LIBROSA_AVAILABLE:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        return audio.astype(np.float32)
    if _SF_AVAILABLE:
        audio, file_sr = sf.read(path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * sr / file_sr))
        return audio.astype(np.float32)
    raise ImportError(
        "Neither librosa nor soundfile is installed. "
        "Run: pip install librosa soundfile"
    )


def process_mm_info(
    conversation: list,
    use_audio_in_video: bool = False,
    sr: int = SAMPLE_RATE,
):
    """
    Extract and load multimodal content from a Qwen conversation dict list.

    Args:
        conversation:       List of role/content message dicts
        use_audio_in_video: Whether to extract audio track from video items
        sr:                 Target sample rate for all loaded audio

    Returns:
        (audios, images, videos)
        - audios: list of float32 numpy arrays, one per <audio> item found
        - images: list (currently unused, reserved for vision tasks)
        - videos: list (currently unused, reserved for video tasks)
    """
    audios = []
    images = []
    videos = []

    for message in conversation:
        content = message.get("content", [])
        if isinstance(content, str):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")

            if item_type == "audio":
                audio_path = item.get("audio", "")
                if audio_path and os.path.exists(audio_path):
                    audios.append(_load_audio(audio_path, sr=sr))
                elif audio_path:
                    # Path provided but file missing — return 1-second silence so
                    # the processor receives a valid tensor shape; validate_audio_path
                    # in scripts will have already warned about the missing file.
                    audios.append(np.zeros(sr, dtype=np.float32))

    return audios, images, videos

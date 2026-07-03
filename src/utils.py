import os
import json
import logging

logger = logging.getLogger(__name__)

def validate_audio_path(audio_path: str) -> bool:
    """Check if audio file exists."""
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}")
        return False
    return True

def load_json(path: str) -> list:
    """Load JSON file safely."""
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: list, path: str) -> None:
    """Save data to JSON file, creating parent directories if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Outputs saved to: {path}")

def set_seed(seed: int = 42) -> None:
    """Set reproducibility seed."""
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to: {seed}")
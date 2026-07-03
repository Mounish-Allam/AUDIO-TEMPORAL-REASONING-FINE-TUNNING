"""One-off script to create/update the results-showcase Hugging Face Space
from the hf_space/ folder. Requires `hf auth login` to have been run already
(uses the locally cached token -- never reads a token from an env var or
argument, so nothing secret passes through this script)."""
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = ROOT / "hf_space"
REPO_ID = "MounishAllam/audio-temporal-reasoning-showcase"

api = HfApi()
who = api.whoami()
print(f"Authenticated as: {who['name']}")

api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True)
print(f"Space repo ready: https://huggingface.co/spaces/{REPO_ID}")

api.upload_folder(
    repo_id=REPO_ID,
    repo_type="space",
    folder_path=str(SPACE_DIR),
    commit_message="Deploy results showcase (scaling curve chart, metrics tables, sample outputs)",
)
print(f"Pushed. Live at: https://huggingface.co/spaces/{REPO_ID}")

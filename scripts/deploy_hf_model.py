"""One-off script to publish the production QLoRA adapter
(checkpoints/final_adapter_3000samples) as a public Hugging Face Model repo.
Requires `hf auth login` to have been run already (uses the locally cached
token -- never reads a token from an env var or argument)."""
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "checkpoints" / "final_adapter_3000samples"
REPO_ID = "MounishAllam/qwen2.5-omni-audio-temporal-qlora"

api = HfApi()
who = api.whoami()
print(f"Authenticated as: {who['name']}")

api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
print(f"Model repo ready: https://huggingface.co/{REPO_ID}")

api.upload_folder(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=str(ADAPTER_DIR),
    commit_message="Publish production QLoRA adapter (3,000-sample run) with model card + real eval results",
)
print(f"Pushed. Live at: https://huggingface.co/{REPO_ID}")

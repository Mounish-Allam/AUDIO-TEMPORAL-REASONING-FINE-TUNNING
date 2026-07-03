import logging
import torch
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from peft import PeftModel

logger = logging.getLogger(__name__)


def load_model_and_processor(
    model_id: str,
    lora_path: str = None,
    gpu_id: int = 0,
    use_4bit: bool = False,
):
    """
    Load Qwen2.5-Omni model and processor, with optional LoRA adapter.

    Args:
        model_id:   HuggingFace model ID  (e.g. "Qwen/Qwen2.5-Omni-7B")
        lora_path:  Path or HF repo ID of a LoRA adapter (None = base model only)
        gpu_id:     CUDA device index; ignored when no GPU is available
        use_4bit:   Load the model in 4-bit NF4 (fits a free 16 GB T4 GPU).
                    Without this the 7B model needs ~16 GB in bf16 alone.

    Returns:
        (model, processor) — model is in eval mode with gradients disabled
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.info("Loading processor from %s", model_id)

    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    load_kwargs = {"device_map": {"": device}}
    if use_4bit:
        logger.info("Loading model in 4-bit NF4 (QLoRA-style quantization)")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # float16 compute: works on T4; bf16 needs Ampere or newer
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    logger.info("Loading model from %s on %s", model_id, device)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_id, **load_kwargs
    )

    if lora_path:
        logger.info("Loading LoRA adapter from %s", lora_path)
        model = PeftModel.from_pretrained(model, lora_path)
        if use_4bit:
            # merge_and_unload is not supported on 4-bit weights;
            # the unmerged adapter works fine for inference.
            logger.info("Keeping adapter unmerged (4-bit model)")
        else:
            model = model.merge_and_unload()
            logger.info("LoRA weights merged and unloaded")

    model.eval()
    logger.info("Model ready on %s", device)
    return model, processor

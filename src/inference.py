import torch
import logging

logger = logging.getLogger(__name__)

def get_model_output(model, processor, conversation: dict) -> str:
    """
    Run inference on a single conversation input.

    Args:
        model:        Loaded Qwen2.5-Omni model
        processor:    Qwen2.5-Omni processor
        conversation: Dict with 'prompt' and 'audios' keys

    Returns:
        Decoded prediction string
    """
    prompt = conversation["prompt"]
    audios = conversation["audios"]

    # Qwen2.5-Omni's processor takes the keyword `audio` (not `audios`)
    inputs = processor(
        text=[prompt],
        audio=audios,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            use_audio_in_video=False,
            do_sample=False,
            stop_strings=["<|im_end|>", "Human:", "User:"],
            tokenizer=processor.tokenizer,
        )

    # Strip input tokens — decode only generated tokens
    generate_ids = out[:, inputs.input_ids.shape[1]:]
    pred = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()

    return pred
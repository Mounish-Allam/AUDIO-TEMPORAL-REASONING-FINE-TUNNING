import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import logging
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

from finetune.dpo_trainer import AudioDPOTrainer
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for audio hallucination reduction")
    parser.add_argument("--model_id",        type=str,   default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--sft_adapter",     type=str,   default=None,
                        help="Path to SFT LoRA adapter to start DPO from")
    parser.add_argument("--preference_data", type=str,   required=True,
                        help="Path to preference JSON file")
    parser.add_argument("--output_dir",      type=str,   default="checkpoints/dpo/")
    parser.add_argument("--num_epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",      type=int,   default=2)
    parser.add_argument("--learning_rate",   type=float, default=5e-5)
    parser.add_argument("--beta",            type=float, default=0.1)
    parser.add_argument("--lora_r",          type=int,   default=16)
    parser.add_argument("--lora_alpha",      type=int,   default=32)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)

    # Load base model
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load SFT adapter if provided, else apply fresh LoRA
    if args.sft_adapter:
        from peft import PeftModel
        logger.info(f"Loading SFT adapter from: {args.sft_adapter}")
        # is_trainable=True is required — the default loads the adapter
        # frozen for inference, and DPO would silently train nothing.
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj",
                "o_proj", "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load preference data
    with open(args.preference_data, 'r') as f:
        preference_data = json.load(f)
    logger.info(f"Loaded {len(preference_data)} preference pairs")

    # Run DPO
    dpo_trainer = AudioDPOTrainer(
        model=model,
        processor=processor,
        preference_data=preference_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
    )

    dpo_trainer.train()

if __name__ == "__main__":
    main()
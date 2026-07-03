import os
import torch
import logging
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

class AudioDPOTrainer:
    """
    DPO (Direct Preference Optimization) trainer for
    Qwen2.5-Omni audio hallucination reduction.

    DPO trains the model to prefer 'chosen' responses
    over 'rejected' responses using paired preference data.

    *** KNOWN LIMITATION — TEXT-ONLY DPO ***
    TRL's DPOTrainer receives only the text prompt here, NOT the audio.
    The model never hears the clip during this stage, so it can only
    learn generic text preferences (e.g. "prefer temporal, specific
    captions"), not audio grounding. Treat this stage as experimental;
    the SFT stage (which DOES train on audio) is the main result.
    A proper multimodal DPO would need a custom loss over audio inputs.

    Expected data format:
    {
        "prompt":   "Question about audio",
        "chosen":   "Accurate faithful answer",
        "rejected": "Hallucinated or wrong answer"
    }
    """

    def __init__(
        self,
        model,
        processor,
        preference_data: list,
        output_dir:    str   = "checkpoints/dpo/",
        num_epochs:    int   = 1,
        batch_size:    int   = 2,
        learning_rate: float = 5e-5,
        beta:          float = 0.1,
        max_length:    int   = 512,
        grad_accum:    int   = 4,
    ):
        self.model      = model
        self.processor  = processor
        self.output_dir = output_dir
        self.beta       = beta  # KL divergence coefficient

        os.makedirs(output_dir, exist_ok=True)

        # Build HuggingFace Dataset from preference pairs
        self.dataset = Dataset.from_list([
            {
                "prompt":   item["prompt"],
                "chosen":   item["chosen"],
                "rejected": item["rejected"],
            }
            for item in preference_data
        ])

        # DPO configuration
        self.dpo_config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            beta=beta,
            max_length=max_length,
            max_prompt_length=256,
            logging_steps=10,
            save_steps=100,
            # bf16 needs an Ampere+ GPU; fall back to fp16 on a free T4
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            remove_unused_columns=False,
        )

    def train(self):
        """Run DPO training."""
        logger.info("Starting DPO preference optimization...")
        logger.info(f"Preference pairs : {len(self.dataset)}")
        logger.info(f"Beta (KL coeff)  : {self.beta}")

        trainer = DPOTrainer(
            model=self.model,
            args=self.dpo_config,
            train_dataset=self.dataset,
            processing_class=self.processor.tokenizer,
        )

        trainer.train()

        # Save final DPO adapter
        final_path = os.path.join(self.output_dir, "final_dpo_adapter")
        self.model.save_pretrained(final_path)
        self.processor.save_pretrained(final_path)
        logger.info(f"DPO adapter saved: {final_path}")

        return trainer
    
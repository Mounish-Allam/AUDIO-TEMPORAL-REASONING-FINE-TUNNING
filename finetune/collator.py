import torch
import logging

logger = logging.getLogger(__name__)


class AudioQACollator:
    """
    Data collator for Audio QA fine-tuning.

    Tokenises a batch of (prompt, audio) pairs and builds labels where
    ONLY the assistant's answer contributes to the loss. Everything
    before the final "<|im_start|>assistant" marker (system prompt,
    audio placeholder tokens, user question) is masked with -100 so the
    model learns to produce answers, not to repeat questions.
    """

    def __init__(self, processor, max_length: int = 1024):
        self.processor  = processor
        self.max_length = max_length
        # Token ids of the assistant-turn marker, used to find where the
        # answer begins inside each tokenised sequence.
        self.assistant_marker = self.processor.tokenizer(
            "<|im_start|>assistant", add_special_tokens=False
        ).input_ids

    def _mask_prompt(self, labels: torch.Tensor) -> torch.Tensor:
        """Set every token up to and including the last assistant marker
        to -100 so the loss only covers the answer."""
        marker = self.assistant_marker
        m = len(marker)
        for row in labels:
            ids = row.tolist()
            start = -1
            for i in range(len(ids) - m, -1, -1):
                if ids[i:i + m] == marker:
                    start = i + m
                    break
            if start >= 0:
                row[:start] = -100
            else:
                # No assistant marker found (e.g. truncated) — skip the
                # whole sample rather than train on the prompt.
                row[:] = -100
                logger.warning("Assistant marker not found; sample masked out")
        return labels

    def __call__(self, batch):
        prompts = [item["prompt"] for item in batch]
        audios  = [item["audio"]  for item in batch]

        # Qwen2.5-Omni's processor takes the keyword `audio` (not `audios`)
        inputs = self.processor(
            text=prompts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels = self._mask_prompt(labels)

        inputs["labels"] = labels
        return inputs

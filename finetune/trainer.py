import os
import torch
import logging
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

logger = logging.getLogger(__name__)

class AudioQATrainer:
    """
    Production-grade trainer for Qwen2.5-Omni LoRA fine-tuning.
    """

    def __init__(
        self,
        model,
        processor,
        dataset,
        collator,
        output_dir:      str   = "checkpoints/",
        num_epochs:      int   = 3,
        batch_size:      int   = 2,
        learning_rate:   float = 2e-4,
        warmup_steps:    int   = 50,
        val_split:       float = 0.1,
        save_steps:      int   = 100,
        logging_steps:   int   = 10,
        gradient_accumulation_steps: int = 4,
    ):
        self.model          = model
        self.processor      = processor
        self.output_dir     = output_dir
        self.num_epochs     = num_epochs
        self.batch_size     = batch_size
        self.learning_rate  = learning_rate
        self.warmup_steps   = warmup_steps
        self.save_steps     = save_steps
        self.logging_steps  = logging_steps
        self.grad_accum     = gradient_accumulation_steps

        os.makedirs(output_dir, exist_ok=True)

        # Train/val split
        val_size   = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Scheduler
        total_steps = (len(self.train_loader) // gradient_accumulation_steps) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def train(self):
        """Main training loop."""
        logger.info("Starting fine-tuning...")
        logger.info(f"Train samples : {len(self.train_dataset)}")
        logger.info(f"Val samples   : {len(self.val_dataset)}")
        logger.info(f"Epochs        : {self.num_epochs}")
        logger.info(f"Batch size    : {self.batch_size}")
        logger.info(f"Learning rate : {self.learning_rate}")

        global_step  = 0
        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}"
            )

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {
                    k: v.to(self.model.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }

                # Forward pass
                outputs = self.model(**batch)
                loss    = outputs.loss / self.grad_accum

                # Backward pass
                loss.backward()
                total_loss += loss.item()

                # Gradient accumulation
                if (step + 1) % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                # Logging
                if global_step % self.logging_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr":   f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })

                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self._save_checkpoint(global_step)

            # Validation
            val_loss = self._validate()
            logger.info(
                f"Epoch {epoch+1} | "
                f"Train Loss: {total_loss/len(self.train_loader):.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("best")
                logger.info(f"Best model saved with val loss: {val_loss:.4f}")

        logger.info("Fine-tuning complete!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

    def _validate(self):
        """Validation loop."""
        self.model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                batch = {
                    k: v.to(self.model.device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                outputs        = self.model(**batch)
                total_val_loss += outputs.loss.item()

        self.model.train()
        return total_val_loss / len(self.val_loader)

    def _save_checkpoint(self, step):
        """Save LoRA adapter checkpoint."""
        path = os.path.join(self.output_dir, f"checkpoint-{step}")
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        logger.info(f"Checkpoint saved: {path}")
# Model Registry

All three QLoRA adapters trained on Qwen2.5-Omni-7B during this project's
scaling study. Numbers are real evaluation results (`outputs/benchmark_report_*.json`),
not simulated. Only **production** is marked deployable — see
[Deployment policy](#deployment-policy) below.

| Run | Folder | Train samples | Epochs | Hallucination (base→SFT) | ROUGE-L (base→SFT) | Weights | Status |
|---|---|---|---|---|---|---|---|
| qlora_v1 | `models/qlora_v1/` | 300 | 1 | 89.0% → 85.0% | 11.5% → 26.5% | lost (overwritten) | archived |
| qlora_v2 | `models/qlora_v2/` | 1,000 | 2 | 90.0% → 78.0% | 11.5% → 31.6% | lost (overwritten) | archived |
| qlora_v3 | `models/production/` | 3,000 | 2 | 86.7% → 77.7% | 12.0% → 31.9% | `checkpoints/final_adapter_3000samples` | **deployable** |

Each run folder contains:
- `metrics.json` — full base vs. SFT benchmark report for that run
- `training_summary.json` — epochs, samples, LoRA hyperparameters, per-epoch
  train/val loss, and weight availability

## Deployment policy

Only `models/production/` (= qlora_v3, the 3,000-sample run) is marked
deployable. It's the only run whose weights still exist, and it has the
lowest hallucination rate and highest ROUGE-L of the three. This is a
deliberate choice, not a default — do not assume "most recently trained"
means "deployable" for future runs; a run must be explicitly promoted here
after review.

## Known caveat affecting all three runs

`scripts/run_finetune.py` saves the model state after the **last** epoch,
not the checkpoint with the best validation loss (`checkpoints/checkpoint-best`
is computed and saved separately but never promoted to the final adapter).
All three runs in this registry overfit between epoch 1 and their last
epoch (validation loss increased while training loss kept dropping), so
every metric here is from a slightly-overfit last-epoch checkpoint rather
than the best one the pipeline already had. See `training_summary.json` in
each folder (`final_saved_checkpoint_matches_best`) and the README's
["Model quality"](README.md#model-quality--what-would-actually-move-the-numbers-next)
section for the fix.

## Weight loss history

`scripts/run_finetune.py` always writes to the same path
(`checkpoints/final_adapter`), so each new training run silently overwrites
the previous adapter's weights unless copied out immediately. This is why
qlora_v1 and qlora_v2 have no surviving weights — only their evaluation
results were saved before the next run overwrote them. qlora_v3's weights
survived because they were copied to `checkpoints/final_adapter_3000samples`
right after training.

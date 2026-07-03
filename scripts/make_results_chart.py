"""Render the real scaling-curve results (README + MODELS.md) as a static
PNG for the README. Reads outputs/benchmark_report_*.json directly -- no
hardcoded numbers -- so re-running this after a new eval regenerates the
chart from the actual JSON.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

RUNS = [
    ("300", "benchmark_report_base.json", "benchmark_report_sft.json"),
    ("1,000", "benchmark_report_base_v2.json", "benchmark_report_sft_v2.json"),
    ("3,000", "benchmark_report_base_v3.json", "benchmark_report_sft_v3.json"),
]

labels, base_halluc, sft_halluc, base_rouge, sft_rouge = [], [], [], [], []
for label, base_file, sft_file in RUNS:
    base = json.loads((OUT / base_file).read_text())
    sft = json.loads((OUT / sft_file).read_text())
    labels.append(f"{label}\nsamples")
    base_halluc.append(base["hallucination_rate"])
    sft_halluc.append(sft["hallucination_rate"])
    base_rouge.append(base["rouge_l"])
    sft_rouge.append(sft["rouge_l"])

BLUE = "#2a78d6"   # categorical slot 1 -- Base
AQUA = "#1baf7a"   # categorical slot 2 -- After QLoRA SFT
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "sans-serif"],
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "xtick.color": INK,
    "ytick.color": MUTED,
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
bar_w = 0.32
x = range(len(labels))

specs = [
    (axes[0], base_halluc, sft_halluc, "Hallucination rate (%)", "lower is better"),
    (axes[1], base_rouge, sft_rouge, "ROUGE-L (%)", "higher is better"),
]

for ax, base_vals, sft_vals, title, sub in specs:
    ax.set_facecolor(SURFACE)
    b1 = ax.bar([i - bar_w / 2 for i in x], base_vals, width=bar_w, color=BLUE, label="Base model")
    b2 = ax.bar([i + bar_w / 2 for i in x], sft_vals, width=bar_w, color=AQUA, label="After QLoRA SFT")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"{title}\n({sub})", fontsize=11, color=INK, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(max(base_vals), max(sft_vals)) * 1.2)

handles, leg_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, leg_labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
           ncol=2, frameon=False, fontsize=10)
fig.suptitle("Real eval results across training-data scale (300 / 1,000 / 3,000 samples)",
             fontsize=12, color=INK, y=1.14)
fig.tight_layout()

out_path = ROOT / "assets" / "scaling_curve.png"
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
print(f"Saved {out_path}")

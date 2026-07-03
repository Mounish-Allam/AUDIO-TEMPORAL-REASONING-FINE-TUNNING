"""Static results showcase for the Audio Temporal Reasoning project.

This Space runs on free CPU hardware and does NOT load Qwen2.5-Omni-7B --
that needs a GPU, which the free tier doesn't have. It shows the real
evaluation results (same numbers as the GitHub README and models/*/metrics.json)
so the project is browsable without needing a GPU machine.

For live inference, see the GitHub repo's Kaggle guide (free T4 GPU).
"""
import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).parent
GITHUB_URL = "https://github.com/Mounish-Allam/AUDIO-TEMPORAL-REASONING-FINE-TUNNING"

RUNS = [
    ("qlora_v1", "300", ROOT / "metrics" / "v1.json"),
    ("qlora_v2", "1,000", ROOT / "metrics" / "v2.json"),
    ("production (v3)", "3,000", ROOT / "metrics" / "v3.json"),
]

def load_results_table():
    rows = []
    for name, samples, path in RUNS:
        data = json.loads(path.read_text())
        base, sft = data["base"], data["sft"]
        rows.append([
            name, samples,
            base["total_samples"],
            f'{base["hallucination_rate"]:.1f}% → {sft["hallucination_rate"]:.1f}%',
            f'{base["rouge_l"]:.1f}% → {sft["rouge_l"]:.1f}%',
            f'{base["temporal_ordering_accuracy"]:.1f}% → {sft["temporal_ordering_accuracy"]:.1f}%',
            f'{base["bert_score"]:.1f}% → {sft["bert_score"]:.1f}%',
        ])
    return rows

# Same three examples curated for the GitHub README's "Sample outputs" section --
# real predictions from outputs/predictions_sft_v3.json, not cherry-picked beyond
# what the README already shows.
SAMPLE_OUTPUTS = [
    ["A man talking as a stream of water trickles in the background",
     "A man speaks as water flows in the background",
     "Accurate paraphrase, no invented sounds"],
    ["A person briefly talks followed quickly by toilet flushing and another voice from another person",
     "A woman speaking and a toilet flushing",
     "Captures the gist; drops the second speaker"],
    ["A person snoring with another man speaking",
     "A man speaks and a pig oinks",
     'Gets the speech right, hallucinates "pig oinks" for the snoring — a real, current failure mode'],
]

THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="emerald", neutral_hue="slate")

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
#header-badges img { display: inline-block; margin: 2px; }
"""

with gr.Blocks(title="Audio Temporal Reasoning — Results Showcase") as demo:
    gr.Markdown(f"""
# 🔊 Audio Temporal Reasoning Pipeline — Results Showcase

QLoRA fine-tuning of **Qwen2.5-Omni-7B** on AudioCaps to reduce audio
hallucination and improve temporal-ordering accuracy in captions.
Full code, training recipe, and honest write-up: **[GitHub repo]({GITHUB_URL})**
""")

    gr.HTML("""
<div id="header-badges">
<img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-2.1%2B-red?logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/PEFT-QLoRA-blueviolet?logo=huggingface">
<img src="https://img.shields.io/badge/Dataset-AudioCaps-yellow">
<img src="https://img.shields.io/badge/Cost-%240-brightgreen">
<img src="https://img.shields.io/badge/License-MIT-lightgrey">
</div>
""")

    gr.Markdown("""
> **This Space shows real, already-computed results.** It does not run the
> model live — Qwen2.5-Omni-7B needs a GPU, and this Space runs on free
> CPU hardware. To run live inference yourself for free, see the repo's
> `KAGGLE_GUIDE.md` (free Kaggle T4 GPU, $0).
""")

    gr.Image(str(ROOT / "assets" / "scaling_curve.png"), show_label=False, container=False)

    gr.Markdown("## 📊 Results across all three training runs")
    gr.Dataframe(
        headers=["Run", "Train samples", "Test samples", "Hallucination (base→SFT)",
                 "ROUGE-L (base→SFT)", "Temporal acc. (base→SFT)", "BERTScore (base→SFT)"],
        value=load_results_table(),
        interactive=False,
        wrap=True,
    )

    gr.Markdown("""
## 🎧 Sample outputs — current production model (3,000-sample run)

Real predictions from `checkpoints/final_adapter_3000samples`, not
cherry-picked — includes a failure case.
""")
    gr.Dataframe(
        headers=["Ground truth", "Production model prediction", "Notes"],
        value=SAMPLE_OUTPUTS,
        interactive=False,
        wrap=True,
    )

    gr.Markdown(f"""
---
Built by Mounish — see the [full README]({GITHUB_URL}#readme) for the
complete failure analysis, metric definitions, and an honest discussion of
what did and didn't improve with more training data.
""")

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)

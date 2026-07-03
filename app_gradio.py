"""
Gradio demo — upload an audio clip, get a temporal description.

Run this on a machine with a GPU (free Kaggle T4 or Colab works):

    python app_gradio.py --use_4bit --lora_path checkpoints/final_adapter

Inside a Kaggle/Colab notebook, `share=True` prints a public link
(e.g. https://xxxxx.gradio.live) that you can open on your phone or
send to an interviewer while the notebook is running.
"""
import argparse
import time

import gradio as gr

from src.model import load_model_and_processor
from src.conversation import build_conversation
from src.inference import get_model_output

DEFAULT_PROMPT = "Describe the temporal order of events in this audio clip."


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for audio temporal reasoning")
    parser.add_argument("--model_id",  type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to your trained LoRA adapter (default: base model)")
    parser.add_argument("--use_4bit",  action="store_true",
                        help="Load model in 4-bit (fits a free 16GB T4 GPU)")
    parser.add_argument("--no_share",  action="store_true",
                        help="Don't create a public gradio.live link")
    args = parser.parse_args()

    print("Loading model (this takes a few minutes the first time)...")
    model, processor = load_model_and_processor(
        args.model_id,
        lora_path=args.lora_path,
        use_4bit=args.use_4bit,
    )
    print("Model ready.")

    def describe(audio_path, prompt):
        if audio_path is None:
            return "Please upload or record an audio clip first."
        prompt = prompt.strip() or DEFAULT_PROMPT

        start = time.perf_counter()
        conversation = build_conversation(processor, prompt, audio_path)
        answer = get_model_output(model, processor, conversation)
        latency_s = time.perf_counter() - start

        return f"{answer}\n\n---\nLatency: {latency_s:.1f} s"

    demo = gr.Interface(
        fn=describe,
        inputs=[
            gr.Audio(type="filepath", label="Audio clip (a few seconds is enough)"),
            gr.Textbox(value=DEFAULT_PROMPT, label="Question"),
        ],
        outputs=gr.Textbox(label="Temporal description"),
        title="Audio Temporal Reasoning — Qwen2.5-Omni",
        description=(
            "Fine-tuned to describe *what you hear and in what order*, "
            "with reduced audio hallucination. "
            f"Adapter: `{args.lora_path or 'none (base model)'}`"
        ),
        flagging_mode="never",
    )

    demo.launch(share=not args.no_share)


if __name__ == "__main__":
    main()

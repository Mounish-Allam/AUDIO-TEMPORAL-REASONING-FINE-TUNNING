import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
from src.model import load_model_and_processor
from src.conversation import build_conversation
from src.inference import get_model_output
from src.utils import validate_audio_path, set_seed

def main():
    parser = argparse.ArgumentParser(description="Single audio inference with Qwen2.5-Omni")
    parser.add_argument("--audio_path",  type=str, required=True, help="Path to .wav audio file")
    parser.add_argument("--prompt",      type=str, required=True, help="Text prompt for the model")
    parser.add_argument("--model_id",    type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--lora_path",   type=str, default=None,
                        help="Path to your trained LoRA adapter (default: base model)")
    parser.add_argument("--gpu_id",      type=int, default=0)
    parser.add_argument("--use_4bit",    action="store_true",
                        help="Load model in 4-bit (fits a free 16GB T4 GPU)")
    args = parser.parse_args()

    set_seed(42)

    if not validate_audio_path(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")

    model, processor = load_model_and_processor(
        args.model_id,
        lora_path=args.lora_path,
        gpu_id=args.gpu_id,
        use_4bit=args.use_4bit,
    )

    conversation = build_conversation(processor, args.prompt, args.audio_path)
    prediction   = get_model_output(model, processor, conversation)

    print(f"\n{'='*50}")
    print(f"Prediction:\n{prediction}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
from qwen_omni_utils import process_mm_info

def build_conversation(processor, prompt: str, audio_path: str) -> dict:
    """
    Build a standard open-ended conversation for audio inference.

    Args:
        processor:   Qwen2.5-Omni processor
        prompt:      Text prompt/question
        audio_path:  Path to .wav audio file

    Returns:
        dict with 'prompt' string and 'audios' list
    """
    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are Qwen, a virtual human developed by the Qwen Team, "
                        "Alibaba Group, capable of perceiving auditory and visual inputs, "
                        "as well as generating text and speech."
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text",  "text": prompt}
            ]
        }
    ]

    prompt_str = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)

    return {"prompt": prompt_str, "audios": audios}


def build_conversation_mcq(processor, question: str, choices: list, audio_path: str) -> dict:
    """
    Build a multiple-choice conversation for audio inference.

    Args:
        processor:   Qwen2.5-Omni processor
        question:    MCQ question string
        choices:     List of answer choice strings
        audio_path:  Path to .wav audio file

    Returns:
        dict with 'prompt' string and 'audios' list
    """
    joined_choices = "\n".join([f"- {choice}" for choice in choices])

    prompt = f"""You are solving a multiple-choice question about an audio clip.

You will be given:
1. A QUESTION about what is happening in the audio.
2. A list of CHOICES.

Your task:
- Carefully listen to the audio.
- Read the QUESTION and CHOICES.
- Choose the single best answer.
- **Important**: Reply with ONLY the exact words and phrases in the choices.
Do NOT include any explanation or extra text.

QUESTION:
{question}

CHOICES:
{joined_choices}

Your answer:
"""

    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are Qwen, a virtual human developed by the Qwen Team, "
                        "Alibaba Group, capable of perceiving auditory and visual inputs, "
                        "as well as generating text and speech."
                    )
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text",  "text": prompt}
            ]
        }
    ]

    prompt_str = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)

    return {"prompt": prompt_str, "audios": audios}
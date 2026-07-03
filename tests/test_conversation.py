import pytest
from unittest.mock import MagicMock, patch
from src.conversation import build_conversation, build_conversation_mcq


class TestConversationBuilder:

    def setup_method(self):
        """Set up mock processor for each test."""
        self.processor = MagicMock()
        self.processor.apply_chat_template.return_value = "<prompt_string>"

    @patch("src.conversation.process_mm_info")
    def test_build_conversation_structure(self, mock_mm):
        """Test basic conversation builds correctly."""
        mock_mm.return_value = (["audio_data"], None, None)

        result = build_conversation(
            processor=self.processor,
            prompt="What do you hear?",
            audio_path="data/audio/test.wav"
        )

        assert "prompt"  in result
        assert "audios"  in result
        assert result["prompt"]  == "<prompt_string>"
        assert result["audios"]  == ["audio_data"]

    @patch("src.conversation.process_mm_info")
    def test_build_conversation_mcq_structure(self, mock_mm):
        """Test MCQ conversation includes choices in prompt."""
        mock_mm.return_value = (["audio_data"], None, None)

        result = build_conversation_mcq(
            processor=self.processor,
            question="What is the dominant sound?",
            choices=["Rain", "Traffic", "Music", "Silence"],
            audio_path="data/audio/test.wav"
        )

        assert "prompt" in result
        assert "audios" in result

        # Verify apply_chat_template was called
        self.processor.apply_chat_template.assert_called_once()

    @patch("src.conversation.process_mm_info")
    def test_mcq_choices_in_prompt(self, mock_mm):
        """Test MCQ prompt contains all choices."""
        mock_mm.return_value = (["audio_data"], None, None)

        choices = ["Option A", "Option B", "Option C"]

        build_conversation_mcq(
            processor=self.processor,
            question="Which sound?",
            choices=choices,
            audio_path="data/audio/test.wav"
        )

        call_args = self.processor.apply_chat_template.call_args
        conversation = call_args[0][0]

        user_content = conversation[1]["content"][1]["text"]
        for choice in choices:
            assert choice in user_content
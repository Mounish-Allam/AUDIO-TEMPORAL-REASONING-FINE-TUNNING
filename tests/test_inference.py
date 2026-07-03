import pytest
import torch
from unittest.mock import MagicMock, patch
from src.inference import get_model_output


class TestInferencePipeline:

    def setup_method(self):
        """Set up mocks for model, processor, and conversation."""
        self.model     = MagicMock()
        self.processor = MagicMock()

        # Mock processor output
        mock_inputs                    = MagicMock()
        mock_inputs.input_ids          = torch.zeros(1, 10, dtype=torch.long)
        mock_inputs.to.return_value    = mock_inputs
        self.processor.return_value    = mock_inputs
        self.mock_inputs               = mock_inputs

        # Mock model generate output
        mock_output                    = torch.zeros(1, 20, dtype=torch.long)
        self.model.generate.return_value = mock_output
        self.model.device              = torch.device("cpu")
        self.model.dtype               = torch.bfloat16

        # Mock decode
        self.processor.batch_decode.return_value = ["Test prediction output"]

        self.conversation = {
            "prompt": "<prompt_string>",
            "audios": ["audio_data"]
        }

    def test_get_model_output_returns_string(self):
        """Test that inference returns a string prediction."""
        result = get_model_output(
            self.model, self.processor, self.conversation
        )
        assert isinstance(result, str)

    def test_model_generate_called(self):
        """Test that model.generate is called once."""
        get_model_output(self.model, self.processor, self.conversation)
        self.model.generate.assert_called_once()

    def test_processor_called_with_list(self):
        """Test processor receives lists for text and audio."""
        get_model_output(self.model, self.processor, self.conversation)
        call_kwargs = self.processor.call_args[1]
        assert isinstance(call_kwargs.get("text"), list)
        assert isinstance(call_kwargs.get("audio"), list)

    def test_output_is_stripped(self):
        """Test that prediction output has no leading/trailing whitespace."""
        self.processor.batch_decode.return_value = ["  padded output  "]
        result = get_model_output(
            self.model, self.processor, self.conversation
        )
        assert result == result.strip()
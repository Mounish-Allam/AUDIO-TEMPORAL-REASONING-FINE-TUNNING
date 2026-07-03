import pytest
import torch
from unittest.mock import patch, MagicMock
from src.model import load_model_and_processor


class TestModelLoading:

    @patch("src.model.Qwen2_5OmniProcessor.from_pretrained")
    @patch("src.model.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained")
    def test_load_base_model(self, mock_model, mock_processor):
        """Test base model loads without LoRA."""
        mock_model.return_value   = MagicMock()
        mock_processor.return_value = MagicMock()
        mock_model.return_value.config = MagicMock()
        mock_model.return_value.eval   = MagicMock()

        model, processor = load_model_and_processor(
            model_id="Qwen/Qwen2.5-Omni-7B",
            lora_path=None
        )

        mock_model.assert_called_once()
        mock_processor.assert_called_once()
        assert model is not None
        assert processor is not None

    @patch("src.model.PeftModel.from_pretrained")
    @patch("src.model.Qwen2_5OmniProcessor.from_pretrained")
    @patch("src.model.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained")
    def test_load_lora_model(self, mock_model, mock_processor, mock_peft):
        """Test LoRA adapter loads correctly."""
        mock_model.return_value     = MagicMock()
        mock_processor.return_value = MagicMock()
        mock_peft.return_value      = MagicMock()
        mock_model.return_value.config = MagicMock()
        mock_model.return_value.eval   = MagicMock()

        model, processor = load_model_and_processor(
            model_id="Qwen/Qwen2.5-Omni-7B",
            lora_path="checkpoints/final_adapter"
        )

        mock_peft.assert_called_once()
        assert model is not None

    def test_model_id_required(self):
        """Test that missing model_id raises an error."""
        with pytest.raises(Exception):
            load_model_and_processor(model_id="")
            
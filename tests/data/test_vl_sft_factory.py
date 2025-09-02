# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoTokenizer

from arctic_training.config.tokenizer import TokenizerConfig
from arctic_training.data.vl_sft_factory import DataCollatorForVisionLanguageCausalLM
from arctic_training.data.vl_sft_factory import VLSFTDataConfig
from arctic_training.data.vl_sft_factory import VLSFTDataFactory
from arctic_training.data.vl_sft_factory import pack_vl_sft_batch
from arctic_training.registry import get_registered_data_factory


def create_dummy_image(size=(224, 224)):
    """Create a dummy RGB image for testing."""
    return Image.new("RGB", size, color="red")


def create_temp_image_file(tmp_path: Path) -> str:
    """Create a temporary image file and return its path."""
    image = create_dummy_image()
    image_path = tmp_path / "test_image.png"
    image.save(image_path)
    return str(image_path)


def create_vl_data_factory(
    model_name: str,
    data_config_kwargs: dict,
    processor=None,
) -> VLSFTDataFactory:
    """Create a VL SFT data factory for testing."""
    data_config_kwargs["type"] = data_config_kwargs.get("type", "vl_sft")

    factory_cls = get_registered_data_factory(data_config_kwargs["type"])
    data_config = VLSFTDataConfig(**data_config_kwargs)

    tokenizer_config = TokenizerConfig(type="huggingface", name_or_path=model_name)

    # Mock processor if not provided
    if processor is None:
        processor = MagicMock()
        processor.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Mock image processing

        def mock_process_images(images=None, return_tensors=None):
            if images:
                batch_size = len(images)
                return {
                    "pixel_values": torch.randn(batch_size, 3, 224, 224),
                    "image_sizes": torch.tensor([[224, 224]] * batch_size)
                }
            return {"pixel_values": torch.empty(0), "image_sizes": torch.empty(0)}
        processor.side_effect = mock_process_images

    dummy_trainer = SimpleNamespace(
        config=SimpleNamespace(
            micro_batch_size=1,
            data=data_config,
            tokenizer=tokenizer_config,
            seed=42,
            gradient_accumulation_steps=1,
            min_iterations=0,
            train_log_iter_interval=0,
        ),
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        processor=processor,
        _set_seeds=lambda seed: None,
    )

    data_factory = factory_cls(trainer=dummy_trainer)
    return data_factory


class TestVLSFTDataConfig:
    """Test VL SFT data configuration."""

    def test_config_defaults(self):
        """Test that VL SFT config has correct defaults."""
        config = VLSFTDataConfig(sources=["test"])
        assert config.max_images_per_sample == 8
        assert config.image_processing_batch_size == 1
        assert config.mask_inputs is True
        assert config.pack_samples is True

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = VLSFTDataConfig(
            sources=["test"],
            max_images_per_sample=4,
            image_processing_batch_size=2,
            mask_inputs=False
        )
        assert config.max_images_per_sample == 4
        assert config.image_processing_batch_size == 2
        assert config.mask_inputs is False


class TestDataCollatorForVisionLanguageCausalLM:
    """Test the vision-language data collator."""

    def test_collator_with_images(self, tmp_path: Path):
        """Test data collation with images."""
        # Create temporary images
        image_path1 = create_temp_image_file(tmp_path)
        image_path2 = create_temp_image_file(tmp_path)

        # Mock processor
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.pad_token_id = 0

        def mock_process(images=None, return_tensors=None):
            if images:
                batch_size = len(images)
                return {
                    "pixel_values": torch.randn(batch_size, 3, 224, 224),
                    "image_sizes": torch.tensor([[224, 224]] * batch_size)
                }
            return {"pixel_values": torch.empty(0)}
        processor.side_effect = mock_process

        # Mock config
        config = MagicMock()
        config.pad_to = "div_length"
        config.div_length = 256
        config.max_length = 512

        collator = DataCollatorForVisionLanguageCausalLM(processor, config)

        # Test instances with images
        instances = [
            {
                "input_ids": [1, 2, 3, 4],
                "labels": [-100, -100, 3, 4],
                "images": [image_path1, image_path2],
            },
            {
                "input_ids": [5, 6, 7],
                "labels": [-100, 6, 7],
                "images": [image_path1],
            }
        ]

        with patch("PIL.Image.open") as mock_open:
            mock_open.return_value = create_dummy_image()

            batch = collator(instances)

            # Check text components
            assert "input_ids" in batch
            assert "labels" in batch
            assert "position_ids" in batch
            assert "packed_sample_seqlens" in batch

            # Check vision components
            assert "pixel_values" in batch
            assert "image_sizes" in batch

            # Verify shapes
            assert batch["input_ids"].shape[0] == 2  # batch size
            assert batch["pixel_values"].shape[0] == 3  # total images across batch

    def test_collator_without_images(self):
        """Test data collation without images."""
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.pad_token_id = 0

        config = MagicMock()
        config.pad_to = "div_length"
        config.div_length = 256

        collator = DataCollatorForVisionLanguageCausalLM(processor, config)

        instances = [
            {
                "input_ids": [1, 2, 3, 4],
                "labels": [-100, -100, 3, 4],
                "images": [],
            }
        ]

        batch = collator(instances)

        # Should have text components but no vision components
        assert "input_ids" in batch
        assert "labels" in batch
        assert "position_ids" in batch
        assert "pixel_values" not in batch or batch["pixel_values"].numel() == 0


class TestPackVLSFTBatch:
    """Test vision-language batch packing functionality."""

    def test_pack_vl_batch_basic(self):
        """Test basic VL batch packing."""
        batch = {
            "input_ids": [[1, 2], [3, 4, 5]],
            "labels": [[-100, 2], [-100, 4, 5]],
            "attention_mask": [[1, 1], [1, 1, 1]],
            "images": [["img1.jpg"], ["img2.jpg", "img3.jpg"]]
        }

        packed = pack_vl_sft_batch(batch, max_length=10, always_max_length=False)

        # Should pack both samples into one
        assert len(packed["input_ids"]) == 1
        assert len(packed["input_ids"][0]) == 5  # 2 + 3 tokens
        assert len(packed["images"][0]) == 3  # All images preserved

    def test_pack_vl_batch_overflow(self):
        """Test VL batch packing when samples exceed max_length."""
        batch = {
            "input_ids": [[1, 2, 3, 4], [5, 6, 7, 8, 9]],
            "labels": [[1, 2, 3, 4], [5, 6, 7, 8, 9]],
            "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1, 1]],
            "images": [["img1.jpg"], ["img2.jpg"]]
        }

        packed = pack_vl_sft_batch(batch, max_length=5, always_max_length=False)

        # Should create two separate samples due to length limit
        assert len(packed["input_ids"]) == 2


class TestVLSFTDataFactory:
    """Test the VL SFT data factory."""

    @pytest.fixture
    def model_name(self):
        return "microsoft/DialoGPT-small"

    def test_factory_registration(self):
        """Test that VL SFT factory is properly registered."""
        factory_cls = get_registered_data_factory("vl_sft")
        assert factory_cls == VLSFTDataFactory

    def test_tokenized_data_processing(self, model_name: str, tmp_path: Path):
        """Test processing of pre-tokenized data."""
        # Create sample tokenized data with images
        image_path = create_temp_image_file(tmp_path)

        dataset = Dataset.from_dict({
            "input_ids": [[1, 2, 3, 4], [5, 6, 7]],
            "labels": [[-100, 2, 3, 4], [-100, 6, 7]],
            "images": [[image_path], []]
        })

        data_factory = create_vl_data_factory(
            model_name=model_name,
            data_config_kwargs={
                "sources": [],
                "cache_dir": tmp_path,
                "pack_samples": False,
                "filter_samples": False
            }
        )

        processed = data_factory.process(dataset)

        # Should add position_ids if not present
        assert "position_ids" in processed.column_names
        assert len(processed) == 2
        assert processed[0]["position_ids"] == [0, 1, 2, 3]

    def test_messages_data_processing(self, model_name: str, tmp_path: Path):
        """Test processing of messages-based data."""
        image_path = create_temp_image_file(tmp_path)

        dataset = Dataset.from_dict({
            "messages": [
                [
                    {"role": "user", "content": "What's in this image?"},
                    {"role": "assistant", "content": "I can see a red square."}
                ]
            ],
            "images": [[image_path]]
        })

        data_factory = create_vl_data_factory(
            model_name=model_name,
            data_config_kwargs={
                "sources": [],
                "cache_dir": tmp_path,
                "pack_samples": False,
                "filter_samples": False
            }
        )

        processed = data_factory.process(dataset)

        # Should have tokenized the messages and preserved images
        assert "input_ids" in processed.column_names
        assert "labels" in processed.column_names
        assert "images" in processed.column_names
        assert len(processed) == 1

    def test_missing_required_columns(self, model_name: str, tmp_path: Path):
        """Test error handling for missing required columns."""
        # Dataset missing images column
        dataset = Dataset.from_dict({
            "input_ids": [[1, 2, 3]],
            "labels": [[1, 2, 3]]
        })

        data_factory = create_vl_data_factory(
            model_name=model_name,
            data_config_kwargs={
                "sources": [],
                "cache_dir": tmp_path
            }
        )

        with pytest.raises(ValueError, match="Dataset must have 'images' column"):
            data_factory.process(dataset)

    def test_tokenize_vl_messages(self, model_name: str):
        """Test the vision-language message tokenization."""
        messages = [
            {"role": "user", "content": "Describe this image"},
            {"role": "assistant", "content": "This is a test image."}
        ]
        images = ["test_image.jpg"]

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "User: Describe this image Assistant: This is a test image."
        tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
            "offset_mapping": [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40)]
        }

        with patch("arctic_training.data.vl_sft_factory.SFTDataFactory.tokenize_messages") as mock_tokenize:
            mock_tokenize.return_value = {
                "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
                "labels": [-100, -100, -100, -100, 5, 6, 7, 8]
            }

            result = VLSFTDataFactory.tokenize_vl_messages(messages, images, tokenizer)

            assert "input_ids" in result
            assert "labels" in result
            assert "images" in result
            assert result["images"] == images

    def test_create_dataloader(self, model_name: str, tmp_path: Path):
        """Test dataloader creation with VL data collator."""
        image_path = create_temp_image_file(tmp_path)

        dataset = Dataset.from_dict({
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "labels": [[-100, 2, 3], [-100, 5, 6]],
            "images": [[image_path], []]
        })

        data_factory = create_vl_data_factory(
            model_name=model_name,
            data_config_kwargs={
                "sources": [],
                "cache_dir": tmp_path,
                "pack_samples": False,
                "filter_samples": False
            }
        )

        dataloader = data_factory.create_dataloader(dataset)

        # Verify dataloader properties
        assert dataloader.batch_size == 1
        assert dataloader.drop_last is True
        assert isinstance(dataloader.collate_fn, DataCollatorForVisionLanguageCausalLM)

    def test_max_images_limit(self, model_name: str, tmp_path: Path):
        """Test that images are limited to max_images_per_sample."""
        # Create more images than the limit
        image_paths = [create_temp_image_file(tmp_path) for _ in range(12)]

        messages = [
            {"role": "user", "content": "What are in these images?"},
            {"role": "assistant", "content": "These are test images."}
        ]

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "conversation"
        tokenizer.return_value = {"input_ids": [1, 2, 3], "labels": [1, 2, 3]}

        with patch("arctic_training.data.vl_sft_factory.SFTDataFactory.tokenize_messages") as mock_tokenize:
            mock_tokenize.return_value = {"input_ids": [1, 2, 3], "labels": [1, 2, 3]}

            result = VLSFTDataFactory.tokenize_vl_messages(messages, image_paths, tokenizer)

            # Should be limited to 8 images
            assert len(result["images"]) == 8

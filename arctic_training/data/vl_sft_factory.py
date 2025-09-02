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

from typing import Dict
from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from transformers import ProcessorMixin

from arctic_training.data.factory import DataFactory
from arctic_training.data.sft_factory import IGNORE_INDEX
from arctic_training.data.sft_factory import SFTDataConfig
from arctic_training.data.sft_factory import filter_dataset_length
from arctic_training.data.sft_factory import pad
from arctic_training.data.utils import DatasetType


class DataCollatorForVisionLanguageCausalLM:
    """Data collator for vision-language tasks supporting LlavaNext models."""

    def __init__(self, processor, config):
        self.processor = processor
        self.config = config

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # Process text components same as regular SFT
        input_ids = [torch.tensor(example["input_ids"]) for example in instances]
        labels = [torch.tensor(example["labels"]) for example in instances]

        if "position_ids" in instances[0]:
            position_ids = [torch.tensor(example["position_ids"]) for example in instances]
            packed_sample_seqlens = [example["packed_sample_seqlens"] for example in instances]
        else:
            position_ids = [torch.tensor(list(range(len(example["input_ids"])))) for example in instances]
            packed_sample_seqlens = [[len(example["input_ids"])] for example in instances]

        # Process images
        images_list = [example.get("images", []) for example in instances]
        pixel_values_list = []
        image_sizes_list = []

        for images in images_list:
            if images and len(images) > 0:
                # Load images from paths and process them
                pil_images = []
                for image_path in images:
                    if isinstance(image_path, str):
                        pil_images.append(Image.open(image_path).convert("RGB"))
                    else:
                        # Assume it's already a PIL image
                        pil_images.append(image_path)

                # Process images with the processor
                processed = self.processor(images=pil_images, return_tensors="pt")
                pixel_values_list.append(processed["pixel_values"])
                if "image_sizes" in processed:
                    image_sizes_list.append(processed["image_sizes"])
            else:
                # No images for this instance - add dummy tensors
                pixel_values_list.append(torch.empty(0))
                image_sizes_list.append(torch.empty(0))

        # Handle padding configuration
        if self.config.pad_to == "max_length":
            pad_kwargs = {"max_seq": self.config.max_length}
        elif self.config.pad_to == "div_length":
            pad_kwargs = {"divisible_by": self.config.div_length}
        else:
            raise ValueError(
                f"Unknown pad_to value: {self.config.pad_to}. Valid values are 'max_length' and 'div_length'."
            )

        # Pad text sequences
        input_ids = pad(input_ids, padding_value=self.processor.tokenizer.pad_token_id, **pad_kwargs)
        labels = pad(labels, padding_value=IGNORE_INDEX, **pad_kwargs)
        position_ids = pad(position_ids, padding_value=0, is_position_id=True, **pad_kwargs)

        # Handle vision components
        batch_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "packed_sample_seqlens": packed_sample_seqlens,
        }

        # Add vision components only when images are present
        if any(len(imgs) > 0 for imgs in images_list):
            # For batching, we need to handle variable number of images per sample
            # Stack pixel_values if they exist and have consistent shapes
            valid_pixel_values = [pv for pv in pixel_values_list if pv.numel() > 0]
            if valid_pixel_values:
                # Concatenate all images from all samples in the batch
                batch_dict["pixel_values"] = torch.cat(valid_pixel_values, dim=0)

                # Handle image_sizes if available
                valid_image_sizes = [is_ for is_ in image_sizes_list if is_.numel() > 0]
                if valid_image_sizes:
                    batch_dict["image_sizes"] = torch.cat(valid_image_sizes, dim=0)

        return batch_dict


class VLSFTDataConfig(SFTDataConfig):
    """Configuration for vision-language SFT data processing."""

    max_images_per_sample: int = 8
    """ Maximum number of images allowed per training sample. """

    image_processing_batch_size: int = 1
    """ Batch size for image processing operations. """


def pack_vl_sft_batch(batch: Dict[str, List], max_length: int, always_max_length: bool) -> Dict[str, List]:
    """Pack vision-language SFT batch similar to text-only version but handling images."""
    keys = ("input_ids", "labels", "position_ids", "packed_sample_seqlens", "attention_mask", "images")
    packed_batch: Dict[str, List] = {k: [] for k in keys}
    current_sample: Dict[str, List] = {k: [] for k in keys if k != "images"}
    current_sample["images"] = []

    def should_flush() -> bool:
        total_len = len(current_sample["input_ids"])
        return total_len > max_length or (not always_max_length and total_len + len(input_ids) > max_length)

    def flush() -> None:
        if len(current_sample["input_ids"]) > 0:
            for k in keys:
                if k == "images":
                    packed_batch[k].append(current_sample[k][:])  # Copy the list
                else:
                    packed_batch[k].append(current_sample[k])
                    current_sample[k] = []
            current_sample["images"] = []

    # Pack multiple samples into one sample
    for input_ids, labels, attention_mask, images in zip(
        batch["input_ids"],
        batch["labels"],
        batch["attention_mask"],
        batch.get("images", [[] for _ in batch["input_ids"]]),
    ):
        if should_flush():
            flush()

        current_sample["input_ids"].extend(input_ids)
        current_sample["labels"].extend(labels)
        current_sample["attention_mask"].extend(attention_mask)
        current_sample["position_ids"].extend(range(len(input_ids)))
        current_sample["packed_sample_seqlens"].extend([len(input_ids)])
        current_sample["images"].extend(images)

    # Add the last example
    flush()

    return packed_batch


def pack_vl_dataset(self, dataset: DatasetType) -> DatasetType:
    """Pack vision-language dataset with image handling."""
    if not self.config.pack_samples:
        return dataset

    batch_size = len(dataset) // self.config.num_proc + 1
    dataset = dataset.shuffle(seed=self.config.seed)
    dataset = dataset.map(
        lambda x: pack_vl_sft_batch(
            x, max_length=self.config.max_length, always_max_length=self.config.always_max_length
        ),
        batched=True,
        batch_size=batch_size,
        num_proc=self.config.num_proc,
        desc="Packing VL dataset",
    )
    if len(dataset) < 1:
        raise ValueError(f"No data left after packing VL dataset samples in {self.__class__.__name__}")
    return dataset


class VLSFTDataFactory(DataFactory):
    """Data factory for vision-language supervised fine-tuning tasks."""

    name = "vl_sft"
    config: VLSFTDataConfig
    callbacks = [
        ("post-load", filter_dataset_length),
        ("post-load", pack_vl_dataset),
    ]

    @property
    def processor(self) -> ProcessorMixin:
        """The processor object used by the Trainer for vision-language models."""
        # Assumes the trainer has a processor attribute for VL models
        return getattr(self.trainer, "processor", self.tokenizer)

    def process(self, dataset: DatasetType) -> DatasetType:
        """Process vision-language dataset with both text and image components."""
        required_columns = ["input_ids", "labels", "images"]

        # Check if we have pre-tokenized data
        has_tokenized_data = all(col in dataset.column_names for col in required_columns[:2])

        if has_tokenized_data:
            # Data is already tokenized, add empty images if not present
            if "images" not in dataset.column_names:
                dataset = dataset.map(
                    lambda example: {**example, "images": []},
                    num_proc=self.config.num_proc,
                    desc="Adding empty images column",
                )

            # Add position_ids if not present
            if "position_ids" not in dataset.column_names:
                dataset = dataset.map(
                    lambda example: {**example, "position_ids": list(range(len(example["input_ids"])))},
                    num_proc=self.config.num_proc,
                    desc="Adding position_ids",
                )

            return dataset
        else:
            # Need to tokenize messages-based data
            if "messages" not in dataset.column_names:
                raise ValueError("Dataset must have either ('input_ids', 'labels', 'images') or 'messages' columns.")

            # Add empty images column if not present (for text-only datasets)
            if "images" not in dataset.column_names:
                dataset = dataset.map(
                    lambda example: {**example, "images": []},
                    num_proc=self.config.num_proc,
                    desc="Adding empty images column for text dataset",
                )

            return dataset.map(
                lambda ex: self.tokenize_vl_messages(
                    ex["messages"],
                    ex.get("images", []),
                    self.tokenizer,
                    mask_inputs=self.config.mask_inputs,
                ),
                remove_columns=[col for col in dataset.column_names if col not in ["images"]],
                num_proc=self.config.num_proc,
                desc="Tokenizing VL messages",
            )

    @staticmethod
    def tokenize_vl_messages(
        messages: List[Dict[str, str]],
        images: List[str],
        tokenizer,
        mask_inputs: bool = True,
    ) -> Dict:
        """Tokenize vision-language messages similar to SFT but preserve image paths."""
        from arctic_training.data.sft_factory import SFTDataFactory

        # Use the existing SFT tokenization logic for text
        tokenized = SFTDataFactory.tokenize_messages(messages, tokenizer, mask_inputs=mask_inputs)

        # Add images to the tokenized result
        tokenized["images"] = images[:8]  # Limit to max 8 images as defined in config

        return tokenized

    def create_dataloader(self, dataset: DatasetType) -> DataLoader:
        """Create a DataLoader with vision-language data collator."""
        return DataLoader(
            dataset,
            collate_fn=DataCollatorForVisionLanguageCausalLM(processor=self.processor, config=self.config),
            batch_size=self.micro_batch_size,
            sampler=DistributedSampler(dataset, num_replicas=self.world_size, rank=self.global_rank),
            num_workers=self.config.dl_num_workers,
            drop_last=True,
        )

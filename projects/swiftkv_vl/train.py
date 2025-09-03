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

from typing import Any, Union

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from arctic_training import HFCheckpointEngine, HFModelFactory, ModelConfig
from arctic_training import SFTTrainer, TrainerConfig, logger
from arctic_training.trainer.sft_trainer import to_device
from projects.swiftkv.models.llava_next import LlavaNextSwiftKVConfig, LlavaNextSwiftKVForConditionalGeneration


class VLSwiftKVModelConfig(ModelConfig):
    """Configuration for VL SwiftKV models."""
    num_key_value_layers: int
    key_value_group_size: int = 1


class VLSwiftKVModelFactory(HFModelFactory):
    """Model factory for VL SwiftKV models."""
    name = "vl_swiftkv_model"
    config: VLSwiftKVModelConfig

    def post_create_config_callback(self, hf_config):
        """Post-process the HF config to add SwiftKV parameters."""
        config_dict = hf_config.to_dict()
        
        model_type = config_dict.get("model_type")
        if model_type == "llava_next_swiftkv":
            # Update text config with SwiftKV parameters
            text_config = config_dict.get("text_config", {})
            text_config["num_key_value_layers"] = self.config.num_key_value_layers
            text_config["key_value_group_size"] = self.config.key_value_group_size
            text_config["swiftkv"] = True
            
            # Create new config with updated parameters
            hf_config = LlavaNextSwiftKVConfig.from_dict(config_dict)
        else:
            raise ValueError(f"Unsupported VL model type: {model_type}")
        
        return hf_config

    def post_create_model_callback(self, model):
        """Initialize SwiftKV parameters in the VL model."""
        if not hasattr(model, 'model') or not hasattr(model.model, 'language_model'):
            raise ValueError("Expected LlavaNext model with language_model component")
        
        language_model = model.model.language_model
        config = language_model.config
        
        # Freeze all teacher parameters initially
        for param in model.parameters():
            param.requires_grad = False
        
        # CPU-friendly initialization (avoid GPU-specific operations)
        self._initialize_swiftkv_parameters_cpu(model, language_model, config)
        
        return model
    
    def _initialize_swiftkv_parameters_cpu(self, model, language_model, config):
        """Initialize SwiftKV parameters using CPU-friendly operations."""
        # Initialize SwiftKV norm
        if hasattr(language_model, 'norm_swiftkv'):
            layer = language_model.layers[config.num_key_value_layers]
            # Simple CPU copy without GPU synchronization
            language_model.norm_swiftkv.weight.data.copy_(layer.input_layernorm.weight.data)
            language_model.norm_swiftkv.weight.requires_grad = True
        
        # Initialize SwiftKV parameters for layers beyond num_key_value_layers
        q_modules = ["q_proj"]
        kv_modules = ["k_proj", "v_proj"]
        
        # Initialize query parameters
        for layer in language_model.layers[config.num_key_value_layers:]:
            attn = layer.self_attn
            for q_module in q_modules:
                if hasattr(attn, f"{q_module}_swiftkv"):
                    teacher_param = getattr(attn, q_module).weight
                    student_param = getattr(attn, f"{q_module}_swiftkv").weight
                    student_param.data.copy_(teacher_param.data)
                    student_param.requires_grad = True
        
        # Initialize KV parameters with group averaging
        for idx, layer in enumerate(language_model.layers[config.num_key_value_layers:]):
            attn = layer.self_attn
            group_idx = idx // config.key_value_group_size
            
            if idx % config.key_value_group_size == 0:
                # This is the first layer in a KV group - initialize SwiftKV params
                kv_attn = attn
                for kv_module in kv_modules:
                    if hasattr(kv_attn, f"{kv_module}_swiftkv"):
                        # Zero out SwiftKV parameters
                        getattr(kv_attn, f"{kv_module}_swiftkv").weight.data.zero_()
                        getattr(kv_attn, f"{kv_module}_swiftkv").weight.requires_grad = True
            
            # Accumulate teacher parameters into SwiftKV parameters
            group_start = config.num_key_value_layers + group_idx * config.key_value_group_size
            kv_layer = language_model.layers[group_start]
            kv_attn = kv_layer.self_attn
            
            for kv_module in kv_modules:
                if hasattr(kv_attn, f"{kv_module}_swiftkv") and hasattr(attn, kv_module):
                    teacher_param = getattr(attn, kv_module).weight
                    student_param = getattr(kv_attn, f"{kv_module}_swiftkv").weight
                    student_param.data.add_(teacher_param.data / config.key_value_group_size)


class VLSwiftKVTrainerConfig(TrainerConfig):
    """Configuration for VL SwiftKV trainer."""
    temperature: float = 1.0
    vision_loss_weight: float = 1.0  # Weight for vision-language loss
    text_loss_weight: float = 1.0    # Weight for text-only loss


class VLSwiftKVTrainer(SFTTrainer):
    """Vision-Language SwiftKV Trainer with distillation."""
    name = "vl_swiftkv"
    config: VLSwiftKVTrainerConfig
    model_factory: VLSwiftKVModelFactory
    checkpoint_engine: Union[HFCheckpointEngine]
    
    def __init__(self, config: VLSwiftKVTrainerConfig):
        super().__init__(config)
        # Initialize processor for vision-language processing
        self.processor = None
        
    def setup_model(self):
        """Setup the VL SwiftKV model and processor."""
        super().setup_model()
        
        # Initialize processor based on the model
        if hasattr(self.model, 'config'):
            model_name = getattr(self.model.config, 'name_or_path', None)
            if model_name:
                try:
                    self.processor = AutoProcessor.from_pretrained(model_name)
                except Exception:
                    logger.warning(f"Could not load processor for {model_name}, using tokenizer only")
                    self.processor = self.tokenizer
            else:
                self.processor = self.tokenizer
        else:
            self.processor = self.tokenizer

    def loss(self, batch: Any) -> torch.Tensor:
        """Compute VL SwiftKV distillation loss."""
        batch = to_device(batch, self.device)
        
        # Determine if this is a vision-language batch
        has_images = 'pixel_values' in batch and batch['pixel_values'].numel() > 0
        
        # Teacher forward pass (SwiftKV disabled)
        with torch.no_grad():
            if hasattr(self.model, 'swiftkv'):
                self.model.swiftkv(False)
            self.model.eval()
            
            # CPU-friendly teacher forward pass
            teacher_outputs = self._forward_batch(batch)
        
        # Student forward pass (SwiftKV enabled)
        if hasattr(self.model, 'swiftkv'):
            self.model.swiftkv(True)
        self.model.train()
        
        student_outputs = self._forward_batch(batch)
        
        # Compute distillation loss
        distill_loss = self.distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            temperature=self.config.temperature
        )
        
        # Apply loss weights based on modality
        if has_images:
            final_loss = distill_loss * self.config.vision_loss_weight
            loss_type = "vision-language"
        else:
            final_loss = distill_loss * self.config.text_loss_weight
            loss_type = "text-only"
        
        # Log losses
        if hasattr(student_outputs, 'loss') and student_outputs.loss is not None:
            student_loss = student_outputs.loss.item()
        else:
            student_loss = 0.0
            
        if hasattr(teacher_outputs, 'loss') and teacher_outputs.loss is not None:
            teacher_loss = teacher_outputs.loss.item()
        else:
            teacher_loss = 0.0
        
        logger.info(
            f"{loss_type} - student loss: {student_loss:.4f}, "
            f"teacher loss: {teacher_loss:.4f}, "
            f"distill loss: {final_loss.item():.4f}"
        )
        
        return final_loss
    
    def _forward_batch(self, batch):
        """Forward pass handling both VL and text-only batches."""
        forward_kwargs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch.get('attention_mask'),
            'labels': batch.get('labels'),
        }
        
        # Add vision inputs if present
        if 'pixel_values' in batch and batch['pixel_values'].numel() > 0:
            forward_kwargs['pixel_values'] = batch['pixel_values']
            if 'image_sizes' in batch and batch['image_sizes'].numel() > 0:
                forward_kwargs['image_sizes'] = batch['image_sizes']
        
        # Remove None values
        forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}
        
        return self.model(**forward_kwargs)

    def distillation_loss(self, student_output, teacher_output, temperature=1.0, dim=-1):
        """Compute KL divergence distillation loss."""
        # Soften the targets
        soft_targets = F.softmax(teacher_output / temperature, dim=dim)
        soft_student = F.log_softmax(student_output / temperature, dim=dim)
        
        # KL divergence loss scaled by temperature squared
        kl_loss = F.kl_div(
            soft_student,
            soft_targets,
            reduction='batchmean',
            log_target=False
        ) * (temperature ** 2)
        
        return kl_loss

    def train_step(self):
        """Single training step optimized for CPU execution."""
        self.model.train()
        
        batch = next(self.train_dataloader_iter)
        
        # Compute loss
        loss = self.loss(batch)
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Optimizer step
        if self.step % self.gradient_accumulation_steps == 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()
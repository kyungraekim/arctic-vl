# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Vision-Language SwiftKV Training Package

This package provides training infrastructure for Vision-Language SwiftKV models,
enabling efficient training of multimodal models using knowledge distillation.

Key Components:
- VLSwiftKVTrainer: Vision-language SwiftKV trainer with distillation
- VLSwiftKVModelFactory: Model factory for VL SwiftKV models  
- VLSwiftKVModelConfig: Configuration for VL SwiftKV models
- VLSwiftKVTrainerConfig: Configuration for VL SwiftKV training

Usage:
    from projects.swiftkv_vl.train import VLSwiftKVTrainer, VLSwiftKVTrainerConfig
    
    trainer = VLSwiftKVTrainer(config)
    trainer.train()
"""

from .train import (
    VLSwiftKVTrainer,
    VLSwiftKVTrainerConfig, 
    VLSwiftKVModelFactory,
    VLSwiftKVModelConfig,
)

__all__ = [
    "VLSwiftKVTrainer",
    "VLSwiftKVTrainerConfig",
    "VLSwiftKVModelFactory", 
    "VLSwiftKVModelConfig",
]

__version__ = "0.1.0"
# Vision-Language SwiftKV Training

This project provides training infrastructure for Vision-Language SwiftKV models, enabling efficient training of multimodal models using knowledge distillation from teacher to student networks.

## Overview

The VL SwiftKV trainer implements:
- **Vision-Language SwiftKV Architecture**: Combines LlavaNext vision-language capabilities with SwiftKV efficiency
- **CPU-Friendly Training**: Optimized for CPU execution without GPU-specific operations
- **Mixed Modality Support**: Handles both vision-language and text-only training samples
- **Knowledge Distillation**: Student model learns from teacher model using KL divergence loss

## Architecture

```
VL SwiftKV Model
├── Vision Tower (CLIP-style encoder)
├── Multimodal Projector (vision → text embedding space)  
└── Language Model (LlamaSwiftKVModel)
    ├── Full KV Layers (0 to num_key_value_layers-1)
    └── SwiftKV Layers (shared KV computation)
```

## Files

- **`train.py`**: Core VL SwiftKV trainer implementation
- **`configs/`**: Training configuration files
- **`example_train.py`**: Example training script with dummy data
- **`README.md`**: This documentation

## Configuration

### Model Configuration
```yaml
model:
  type: vl_swiftkv
  name_or_path: projects.swiftkv.models.llava_next.create_small_llava_next_swiftkv_config
  num_key_value_layers: 2    # First N layers with full KV
  key_value_group_size: 1    # KV sharing group size
  attn_implementation: eager # CPU-friendly attention
```

### Data Configuration
```yaml
data:
  type: vl_sft
  sources:
    - HuggingFaceM4/VQAv2        # Vision QA dataset
    - lmsys/lmsys-chat-1m        # Text-only dataset
  max_images_per_sample: 2       # Limit images per sample
  image_processing_batch_size: 1 # CPU-friendly batch size
  max_length: 512                # Sequence length limit
```

### CPU Training Settings
```yaml
micro_batch_size: 1
gradient_accumulation_steps: 4
deepspeed:
  zero_optimization:
    stage: 0                    # Disable ZeRO for CPU
```

## Usage

### Basic Training Example
```bash
# Run with default small configuration
python example_train.py

# Use custom configuration
python example_train.py --config configs/my-config.yaml

# Dry run to test setup
python example_train.py --dry-run
```

### Integration with Arctic Training
```bash
# Use with arctic_training CLI
arctic_training projects/swiftkv_vl/configs/small-llava-next-swiftkv-cpu.yaml
```

## Key Features

### 🧠 **SwiftKV Integration**
- Efficient key-value computation sharing
- Dynamic SwiftKV toggle for teacher/student modes
- Group-based KV parameter sharing

### 👁️ **Vision-Language Support**
- Processes both multimodal and text-only batches
- Automatic image preprocessing and batching
- Mixed dataset training capabilities

### 💻 **CPU-Optimized**
- No GPU-specific operations (flash attention, etc.)
- Memory-efficient parameter initialization
- Single-process data loading

### 📊 **Training Monitoring**
- Separate loss tracking for VL vs text-only samples
- Student/teacher loss comparison
- Distillation loss monitoring

## Training Process

1. **Teacher Forward Pass**: Model runs with SwiftKV disabled (full computation)
2. **Student Forward Pass**: Model runs with SwiftKV enabled (efficient computation)
3. **Distillation Loss**: KL divergence between teacher and student logits
4. **Weighted Loss**: Different weights for vision-language vs text-only samples

## Loss Function

```python
# KL divergence distillation loss
kl_loss = F.kl_div(
    F.log_softmax(student_logits / temperature, dim=-1),
    F.softmax(teacher_logits / temperature, dim=-1),
    reduction='batchmean'
) * (temperature ** 2)

# Weighted by modality
final_loss = kl_loss * (vision_loss_weight if has_images else text_loss_weight)
```

## Model Sizes

### Small Configuration (Testing)
- **Text Model**: 64d hidden, 4 layers, 2 attention heads
- **Vision Model**: 64d hidden, 2 layers  
- **Total Parameters**: ~4.4M
- **SwiftKV Layers**: Last 2 layers (50% reduction)

### Custom Configurations
Modify `create_small_llava_next_swiftkv_config()` or create new config functions for different model sizes.

## Requirements

- `torch` (CPU support)
- `transformers` 
- `datasets`
- `PIL` (for image processing)
- `arctic_training` framework

## CPU vs GPU

This implementation is designed for CPU compatibility:
- ✅ **CPU**: Fully supported, optimized for CPU execution
- 🔄 **GPU**: Can be adapted by removing CPU-specific constraints

For GPU training:
1. Enable DeepSpeed ZeRO optimization
2. Use flash attention (`attn_implementation: flash_attention_2`)
3. Increase batch sizes and sequence lengths
4. Enable gradient checkpointing

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `max_length`, `micro_batch_size`, or `max_images_per_sample`
2. **Slow Training**: Increase `gradient_accumulation_steps`, reduce `num_proc`
3. **Model Loading**: Ensure all SwiftKV models are properly registered

### Debug Mode
```bash
python example_train.py --dry-run --num-samples 5
```

## Contributing

To extend the VL SwiftKV trainer:

1. **New Model Types**: Add support in `VLSwiftKVModelFactory.post_create_config_callback()`
2. **New Loss Functions**: Extend `VLSwiftKVTrainer.loss()` method
3. **Data Sources**: Use VL SFT factory's data source system
4. **CPU Optimizations**: Focus on memory efficiency and parameter initialization

## Examples

See `example_train.py` for a complete working example with dummy data generation and CPU-optimized training setup.
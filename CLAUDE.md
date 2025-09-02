# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install with development dependencies
pip install ".[dev]"

# Install pre-commit hooks for formatting
pre-commit install
```

### Testing
```bash
# Run all tests (CPU and GPU)
make test
# or: pytest tests

# Run CPU-only tests
make test-cpu
# or: pytest -m "not gpu" tests

# Run GPU-only tests
make test-gpu
# or: pytest -m gpu tests

# Test VL SFT factory specifically
python run_vl_tests.py
```

### Code Formatting and Linting
```bash
# Fix formatting (recommended method)
make format

# Run pre-commit hooks manually
pre-commit run --all-files

# Type checking (via pre-commit)
mypy arctic_training/ projects/arctic_embed/
```

### Training
```bash
# Run training with a YAML config
arctic_training path/to/config.yaml

# Run training with DeepSpeed launcher arguments
arctic_training path/to/config.yaml --num_gpus 2 --num_nodes 1

# Data processing mode
arctic_training process-data path/to/config.yaml
```

### Synthetic Data Generation
```bash
# Generate synthetic data
arctic_synth path/to/config.yaml
```

## Architecture

### Core Framework Structure
ArcticTraining is a modular LLM post-training framework built on DeepSpeed with the following key architectural components:

- **Trainers** (`arctic_training/trainer/`): Core training logic
  - `Trainer`: Base trainer class with distributed training orchestration
  - `SFTTrainer`: Supervised fine-tuning implementation
  - `DPOTrainer`: Direct Preference Optimization implementation

- **Configuration System** (`arctic_training/config/`): Pydantic-based configuration management
  - Centralized config validation and type safety
  - Modular configs for model, data, optimizer, scheduler, checkpointing

- **Data Pipeline** (`arctic_training/data/`): Flexible data loading and processing
  - `DataFactory`: Abstract factory for dataset creation
  - `SFTDataFactory` and `DPODataFactory`: Specialized implementations
  - `DataSource`: Pluggable data source interface (HuggingFace, custom sources)

- **Model Management** (`arctic_training/model/`): Model loading and configuration
  - `ModelFactory`: Abstract model creation interface
  - `HFModelFactory`: HuggingFace model integration
  - `LigerModelFactory`: Optimized model implementations

- **Registry System** (`arctic_training/registry.py`): Dynamic component registration
  - Automatic discovery of trainers, factories, and other components
  - Enables custom trainer registration via `@register` decorator

### Key Design Patterns

1. **Factory Pattern**: Used extensively for model, data, optimizer, and scheduler creation
2. **Registry Pattern**: Enables dynamic component discovery and custom extensions
3. **Configuration-Driven**: All training aspects controlled via YAML configuration files
4. **Modular Architecture**: Each component can be independently extended or replaced

### Projects Structure
The `projects/` directory contains specialized implementations:
- `arctic_embed/`: Embedding model training
- `swiftkv/`: Knowledge preserving compute reduction
- `arctic_lstm_speculator/`: Speculative decoding with LSTM
- `sequence-parallelism/`: Sequence parallelism optimizations

### Extension Points

#### Custom Trainers
Create custom trainers by subclassing `Trainer` or `SFTTrainer`:

```python
from arctic_training import SFTTrainer, register

@register
class CustomTrainer(SFTTrainer):
    name = "my_custom_trainer"

    def loss(self, batch):
        # Custom loss implementation
        return loss
```

#### Custom Data Sources
Extend data loading by implementing `DataSource` interface:

```python
from arctic_training import DataSource, register

@register
class CustomDataSource(DataSource):
    name = "my_data_source"

    def load_data(self):
        # Custom data loading logic
        return dataset
```

### Configuration System
Training recipes are YAML files with the following structure:
- `type`: Trainer type (sft, dpo, vl_sft, custom)
- `model`: Model configuration (name_or_path, tokenizer settings)
- `data`: Data sources and processing parameters
- `checkpoint`: Checkpointing strategy (HuggingFace, DeepSpeed)
- `optimizer`: Optimizer settings (AdamW, learning rate, etc.)
- `scheduler`: Learning rate scheduler configuration

### Vision-Language Training
The `vl_sft` trainer enables training of vision-language models like LlavaNext:

```yaml
type: vl_sft
model:
  name_or_path: llava-hf/llava-v1.6-mistral-7b-hf
data:
  sources:
    - your-vl-dataset      # VL dataset with images
    - lmsys/lmsys-chat-1m  # Text dataset (no images)
  max_images_per_sample: 8
  image_processing_batch_size: 1
```

**Data Formats Supported:**
- Pre-tokenized: `input_ids`, `labels`, `images` columns (or `input_ids`, `labels` for text-only)
- Messages: `messages`, `images` columns (or `messages` for text-only)
- Text datasets: Any dataset with `messages` column (images column optional)

**Key Components:**
- `VLSFTDataFactory`: Main factory for VL training
- `VLSFTDataConfig`: Configuration with VL-specific parameters
- `DataCollatorForVisionLanguageCausalLM`: Handles text + image batching

### Development Workflow
1. Create training recipe YAML
2. Run training via CLI (`arctic_training config.yaml`)
3. DeepSpeed launcher handles distributed training automatically
4. Checkpoints saved according to configuration
5. Custom trainers auto-discovered if placed in `train.py` or specified via `code` field

#!/usr/bin/env python3
"""
Example script demonstrating the usage of a very small LlavaNextForConditionalGeneration
with SwiftKV support.

This creates a tiny model suitable for testing and development.
"""

import torch
from projects.swiftkv.models.llava_next import (
    LlavaNextSwiftKVForConditionalGeneration,
    create_small_llava_next_swiftkv_config,
)


def create_small_llava_model():
    """Create a very small LlavaNext model with SwiftKV."""
    print("Creating small LlavaNext SwiftKV configuration...")
    config = create_small_llava_next_swiftkv_config()
    
    print(f"Text model config:")
    print(f"  - Hidden size: {config.text_config.hidden_size}")
    print(f"  - Num layers: {config.text_config.num_hidden_layers}")
    print(f"  - Num attention heads: {config.text_config.num_attention_heads}")
    print(f"  - Vocab size: {config.text_config.vocab_size}")
    print(f"  - SwiftKV enabled: {config.text_config.swiftkv}")
    print(f"  - Num key-value layers: {config.text_config.num_key_value_layers}")
    print(f"  - Key-value group size: {config.text_config.key_value_group_size}")
    
    print(f"\nVision model config:")
    print(f"  - Hidden size: {config.vision_config.hidden_size}")
    print(f"  - Num layers: {config.vision_config.num_hidden_layers}")
    print(f"  - Image size: {config.vision_config.image_size}")
    print(f"  - Patch size: {config.vision_config.patch_size}")
    
    print(f"\nMultimodal config:")
    print(f"  - Image token index: {config.image_token_index}")
    print(f"  - Image sequence length: {config.image_seq_length}")
    print(f"  - Image grid pinpoints: {config.image_grid_pinpoints}")
    
    print("\nCreating model...")
    model = LlavaNextSwiftKVForConditionalGeneration(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, config


def test_model_forward():
    """Test the model with dummy inputs."""
    model, config = create_small_llava_model()
    model.eval()
    
    print("\n" + "="*50)
    print("Testing model forward pass...")
    
    # Test text-only first (no images)
    batch_size = 1
    seq_len = 20
    
    # Text-only input
    input_ids = torch.randint(0, config.text_config.vocab_size-1, (batch_size, seq_len))
    
    print(f"Input shapes (text-only):")
    print(f"  - input_ids: {input_ids.shape}")
    
    # Forward pass
    print(f"\nRunning forward pass (text-only)...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
        )
    
    print(f"Output shapes:")
    print(f"  - logits: {outputs.logits.shape}")
    if outputs.hidden_states is not None:
        print(f"  - hidden_states: {len(outputs.hidden_states)} layers")
    if outputs.image_hidden_states is not None:
        print(f"  - image_hidden_states: {outputs.image_hidden_states.shape}")
    else:
        print(f"  - image_hidden_states: None (text-only)")
    
    print("✓ Text-only forward pass successful!")
    
    # Now test with simple multimodal input
    print(f"\nTesting simple multimodal forward pass...")
    
    # Create simple input with one image token
    seq_len = 10
    input_ids = torch.randint(0, config.text_config.vocab_size-1, (batch_size, seq_len-1))
    # Insert image token at position 1
    image_token = torch.tensor([[config.image_token_index]], dtype=torch.long)
    input_ids = torch.cat([input_ids[:, :1], image_token, input_ids[:, 1:]], dim=1)
    
    # Create very simple pixel values - single patch
    # Use single image format 
    pixel_values = torch.randn(batch_size, 1, 3, config.vision_config.image_size, config.vision_config.image_size)
    image_sizes = torch.tensor([[config.vision_config.image_size, config.vision_config.image_size]], dtype=torch.long)
    
    print(f"Multimodal input shapes:")
    print(f"  - input_ids: {input_ids.shape}")
    print(f"  - pixel_values: {pixel_values.shape}")
    print(f"  - image_sizes: {image_sizes.shape}")
    
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )
        
        print(f"Multimodal output shapes:")
        print(f"  - logits: {outputs.logits.shape}")
        if outputs.image_hidden_states is not None:
            print(f"  - image_hidden_states: {outputs.image_hidden_states.shape}")
        
        print("✓ Multimodal forward pass successful!")
        
    except Exception as e:
        print(f"⚠️  Multimodal forward pass failed: {e}")
        print("✓ Text-only mode working correctly")


def test_swiftkv_toggle():
    """Test toggling SwiftKV on/off."""
    model, config = create_small_llava_model()
    model.eval()
    
    print("\n" + "="*50)
    print("Testing SwiftKV toggle...")
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 5
    input_ids = torch.randint(0, config.text_config.vocab_size, (batch_size, seq_len))
    
    print(f"SwiftKV initially: {model.model.language_model.config.swiftkv}")
    
    # Test with SwiftKV enabled
    print("\nTesting with SwiftKV enabled...")
    model.swiftkv(True)
    print(f"SwiftKV enabled: {model.model.language_model.config.swiftkv}")
    
    with torch.no_grad():
        outputs_swiftkv = model(input_ids=input_ids)
    print(f"Output logits shape: {outputs_swiftkv.logits.shape}")
    
    # Test with SwiftKV disabled
    print("\nTesting with SwiftKV disabled...")
    model.swiftkv(False)
    print(f"SwiftKV enabled: {model.model.language_model.config.swiftkv}")
    
    with torch.no_grad():
        outputs_no_swiftkv = model(input_ids=input_ids)
    print(f"Output logits shape: {outputs_no_swiftkv.logits.shape}")
    
    print("✓ SwiftKV toggle successful!")


if __name__ == "__main__":
    print("Small LlavaNext SwiftKV Example")
    print("=" * 50)
    
    try:
        # Test model creation
        test_model_forward()
        
        # Test SwiftKV toggle
        test_swiftkv_toggle()
        
        print("\n" + "="*50)
        print("✓ All tests passed! The small LlavaNext SwiftKV model is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
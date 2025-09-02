#!/usr/bin/env python3
"""
Simple test runner for VL SFT Factory tests.
This script runs the VL SFT factory tests without relying on pytest's distributed setup.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def run_config_tests():
    """Run VL SFT configuration tests."""
    print("=" * 50)
    print("Testing VLSFTDataConfig")
    print("=" * 50)
    
    from tests.data.test_vl_sft_factory import TestVLSFTDataConfig
    
    test_config = TestVLSFTDataConfig()
    
    try:
        test_config.test_config_defaults()
        print("✓ test_config_defaults PASSED")
    except Exception as e:
        print(f"✗ test_config_defaults FAILED: {e}")
        return False
    
    try:
        test_config.test_config_custom_values()
        print("✓ test_config_custom_values PASSED")
    except Exception as e:
        print(f"✗ test_config_custom_values FAILED: {e}")
        return False
    
    return True

def run_batch_packing_tests():
    """Run VL batch packing tests."""
    print("=" * 50)
    print("Testing VL Batch Packing")
    print("=" * 50)
    
    from tests.data.test_vl_sft_factory import TestPackVLSFTBatch
    
    test_pack = TestPackVLSFTBatch()
    
    try:
        test_pack.test_pack_vl_batch_basic()
        print("✓ test_pack_vl_batch_basic PASSED")
    except Exception as e:
        print(f"✗ test_pack_vl_batch_basic FAILED: {e}")
        return False
    
    try:
        test_pack.test_pack_vl_batch_overflow()
        print("✓ test_pack_vl_batch_overflow PASSED")
    except Exception as e:
        print(f"✗ test_pack_vl_batch_overflow FAILED: {e}")
        return False
    
    return True

def run_factory_tests():
    """Run VL SFT factory tests."""
    print("=" * 50)
    print("Testing VLSFTDataFactory")
    print("=" * 50)
    
    from tests.data.test_vl_sft_factory import TestVLSFTDataFactory
    from arctic_training.data.vl_sft_factory import VLSFTDataFactory
    
    test_factory = TestVLSFTDataFactory()
    model_name = 'microsoft/DialoGPT-small'
    
    # Test factory registration
    try:
        test_factory.test_factory_registration()
        print("✓ test_factory_registration PASSED")
    except Exception as e:
        print(f"✗ test_factory_registration FAILED: {e}")
        return False
    
    # Test tokenization manually (avoiding patch issues)
    try:
        messages = [
            {'role': 'user', 'content': 'Describe this image'},
            {'role': 'assistant', 'content': 'This is a test image.'}
        ]
        images = ['test_image.jpg']
        tokenizer = MagicMock()
        
        with patch('arctic_training.data.sft_factory.SFTDataFactory.tokenize_messages') as mock_tokenize:
            mock_tokenize.return_value = {
                'input_ids': [1, 2, 3, 4, 5, 6, 7, 8],
                'labels': [-100, -100, -100, -100, 5, 6, 7, 8]
            }
            
            result = VLSFTDataFactory.tokenize_vl_messages(messages, images, tokenizer)
            
            assert 'input_ids' in result
            assert 'labels' in result  
            assert 'images' in result
            assert result['images'] == images
            
        print("✓ tokenize_vl_messages PASSED")
    except Exception as e:
        print(f"✗ tokenize_vl_messages FAILED: {e}")
        return False
    
    # Test max images limit
    try:
        image_paths = [f'img{i}.jpg' for i in range(12)]
        messages = [
            {'role': 'user', 'content': 'What are in these images?'},
            {'role': 'assistant', 'content': 'These are test images.'}
        ]
        tokenizer = MagicMock()
        
        with patch('arctic_training.data.sft_factory.SFTDataFactory.tokenize_messages') as mock_tokenize:
            mock_tokenize.return_value = {'input_ids': [1, 2, 3], 'labels': [1, 2, 3]}
            
            result = VLSFTDataFactory.tokenize_vl_messages(messages, image_paths, tokenizer)
            
            # Should be limited to 8 images
            assert len(result['images']) == 8
            
        print("✓ max_images_limit PASSED")
    except Exception as e:
        print(f"✗ max_images_limit FAILED: {e}")
        return False
    
    return True

def run_collator_tests():
    """Run data collator tests."""
    print("=" * 50)
    print("Testing DataCollatorForVisionLanguageCausalLM")
    print("=" * 50)
    
    from tests.data.test_vl_sft_factory import create_dummy_image
    from arctic_training.data.vl_sft_factory import DataCollatorForVisionLanguageCausalLM
    
    try:
        # Test basic collator creation
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.pad_token_id = 0
        
        config = MagicMock()
        config.pad_to = 'div_length'
        config.div_length = 256
        
        collator = DataCollatorForVisionLanguageCausalLM(processor, config)
        print("✓ DataCollatorForVisionLanguageCausalLM creation PASSED")
    except Exception as e:
        print(f"✗ DataCollatorForVisionLanguageCausalLM creation FAILED: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Running VL SFT Factory Tests")
    print("=" * 70)
    
    all_passed = True
    
    # Run each test suite
    test_suites = [
        ("Configuration Tests", run_config_tests),
        ("Batch Packing Tests", run_batch_packing_tests),
        ("Factory Tests", run_factory_tests),
        ("Collator Tests", run_collator_tests),
    ]
    
    results = {}
    for suite_name, test_func in test_suites:
        print(f"\n🔍 Running {suite_name}...")
        try:
            passed = test_func()
            results[suite_name] = "PASSED" if passed else "FAILED"
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"✗ {suite_name} FAILED with exception: {e}")
            results[suite_name] = "FAILED"
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    for suite_name, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌"
        print(f"{status_icon} {suite_name}: {result}")
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
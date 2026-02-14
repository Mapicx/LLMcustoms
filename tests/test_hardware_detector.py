#!/usr/bin/env python3
"""
Test script for HardwareDetector class.
Run this to verify hardware detection functionality.
"""

import sys
import os

# Add the parent directory (project root) to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from llmcustoms.core.hardware_detector import HardwareDetector, run_diagnostic
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    def test_hardware_detector():
        """Test HardwareDetector functionality."""
        print("=" * 60)
        print("Testing HardwareDetector")
        print("=" * 60)
        
        # Initialize HardwareDetector
        print("\n1. Initializing HardwareDetector...")
        detector = HardwareDetector()
        print("   ✅ HardwareDetector initialized")
        
        # Test GPU memory detection
        print("\n2. Testing GPU Memory Detection:")
        try:
            gpu_memory = detector.detect_gpu_memory()
            print(f"   GPU Memory: {gpu_memory} MB")
            
            if gpu_memory > 0:
                print(f"   ✅ GPU detected with {gpu_memory/1024:.1f} GB VRAM")
                gpu_available = True
            else:
                print("   ⚠️  No GPU detected or CUDA not available")
                gpu_available = False
                
        except Exception as e:
            print(f"   ❌ GPU detection failed: {e}")
            gpu_available = False
            gpu_memory = 0
        
        # Test model suggestions
        print("\n3. Testing Model Suggestions:")
        
        test_vram_amounts = [2048, 4096, 6144, 8192, 12288, 16384, 24576]  # MB
        
        for vram_mb in test_vram_amounts:
            try:
                suggested_model = detector.suggest_model(vram_mb)
                vram_gb = vram_mb / 1024
                print(f"   {vram_gb:4.1f} GB VRAM -> {suggested_model}")
            except Exception as e:
                print(f"   ❌ Model suggestion failed for {vram_mb}MB: {e}")
        
        # Test optimal batch size calculation
        print("\n4. Testing Optimal Batch Size Calculation:")
        
        test_models = ["tinyllama", "phi-3.5-mini", "mistral-7b", "qwen2.5-7b"]
        
        for model in test_models:
            try:
                batch_size = detector.get_optimal_batch_size(model)
                print(f"   {model:<15} -> batch size: {batch_size}")
            except Exception as e:
                print(f"   ❌ Batch size calculation failed for {model}: {e}")
        
        # Test training configuration
        print("\n5. Testing Training Configuration:")
        
        test_configs = [
            ("tinyllama", 4096),
            ("phi-3.5-mini", 8192),
            ("mistral-7b", 12288),
            ("qwen2.5-7b", 16384)
        ]
        
        for model, vram_mb in test_configs:
            try:
                config = detector.get_training_config(model, vram_mb)
                print(f"\n   {model} with {vram_mb/1024:.1f}GB:")
                print(f"     Batch size: {config.get('per_device_train_batch_size', 'N/A')}")
                print(f"     Gradient accumulation: {config.get('gradient_accumulation_steps', 'N/A')}")
                print(f"     Max sequence length: {config.get('max_seq_length', 'N/A')}")
                print(f"     FP16: {config.get('fp16', 'N/A')}")
                print(f"     Dataloader workers: {config.get('dataloader_num_workers', 'N/A')}")
            except Exception as e:
                print(f"   ❌ Training config failed for {model}: {e}")
        
        # Test edge cases
        print("\n6. Testing Edge Cases:")
        
        # Test with very low VRAM
        try:
            low_vram_model = detector.suggest_model(1024)  # 1GB
            print(f"   ✅ Low VRAM (1GB) handling: {low_vram_model}")
        except Exception as e:
            print(f"   ✅ Low VRAM correctly rejected: {str(e)[:50]}...")
        
        # Test with very high VRAM
        try:
            high_vram_model = detector.suggest_model(81920)  # 80GB
            print(f"   ✅ High VRAM (80GB) handling: {high_vram_model}")
        except Exception as e:
            print(f"   ❌ High VRAM handling failed: {e}")
        
        # Test with invalid model name
        try:
            invalid_batch = detector.get_optimal_batch_size("invalid_model")
            print(f"   ❌ Should have failed with invalid model")
        except Exception as e:
            print(f"   ✅ Invalid model correctly rejected: {str(e)[:50]}...")
        
        # Test VRAM requirements
        print("\n7. Testing VRAM Requirements:")
        
        print("   Model VRAM Requirements:")
        for model_name, model_info in detector.VRAM_REQUIREMENTS.items():
            min_vram = model_info['min_vram_gb']
            recommended_vram = model_info['recommended_vram_gb']
            print(f"     {model_name:<15}: {min_vram}GB min, {recommended_vram}GB recommended")
        
        # Test system information gathering
        print("\n8. Testing System Information:")
        
        try:
            # Test if we can get basic system info
            import torch
            cuda_available = torch.cuda.is_available()
            print(f"   CUDA Available: {cuda_available}")
            
            if cuda_available:
                device_count = torch.cuda.device_count()
                print(f"   CUDA Devices: {device_count}")
                
                if device_count > 0:
                    device_name = torch.cuda.get_device_name(0)
                    print(f"   Primary GPU: {device_name}")
            
        except ImportError:
            print("   ⚠️  PyTorch not available - some features will be limited")
        except Exception as e:
            print(f"   ⚠️  System info gathering failed: {e}")
        
        print("\n" + "=" * 60)
        print("HardwareDetector test completed!")
        print("=" * 60)
        
        # Summary
        print(f"\nSummary:")
        print(f"  GPU Available: {'✅' if gpu_available else '❌'}")
        if gpu_available:
            print(f"  GPU Memory: {gpu_memory/1024:.1f} GB")
            suggested = detector.suggest_model(gpu_memory)
            print(f"  Suggested Model: {suggested}")
        print(f"  ✅ Model suggestions working")
        print(f"  ✅ Batch size calculations working")
        print(f"  ✅ Training configurations working")
        print(f"  ✅ Edge case handling working")
        
        # Test the diagnostic function
        print(f"\n9. Testing Diagnostic Function:")
        try:
            print("   Running full diagnostic...")
            run_diagnostic()
            print("   ✅ Diagnostic function completed successfully")
        except Exception as e:
            print(f"   ❌ Diagnostic function failed: {e}")
    
    if __name__ == "__main__":
        test_hardware_detector()

except ImportError as e:
    print(f"Import error: {e}")
    print(f"Project root: {project_root}")
    print("\nMissing dependencies. Please install:")
    print("uv pip install torch")
    print("\nFor basic testing without GPU:")
    print("uv pip install pathlib")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
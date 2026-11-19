#!/usr/bin/env python3
"""
XPU Optimization Validation Script for HeartMuLa
Tests all optimizations and reports system readiness
PyTorch 2.10+ has native XPU support (IPEX discontinued)
"""

import sys
import os

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def check_mark(condition):
    return "✓" if condition else "✗"

def test_pytorch_xpu():
    """Test PyTorch XPU availability"""
    try:
        import torch
        has_xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()
        if has_xpu:
            device_name = torch.xpu.get_device_name(0)
            total_mem = torch.xpu.get_device_properties(0).total_memory / (1024**3)
            print(f"{check_mark(True)} PyTorch XPU: Available (Native Support)")
            print(f"  Device: {device_name}")
            print(f"  Memory: {total_mem:.2f} GB")
            print(f"  PyTorch: {torch.__version__}")
            return True, torch.__version__
        else:
            print(f"{check_mark(False)} PyTorch XPU: Not available")
            return False, torch.__version__
    except Exception as e:
        print(f"{check_mark(False)} PyTorch: Import error - {e}")
        return False, None

def test_native_xpu_features():
    """Test native XPU features"""
    try:
        import torch
        print(f"{check_mark(True)} Native XPU Support: PyTorch {torch.__version__}")
        print(f"  torch.compile: {hasattr(torch, 'compile')}")
        print(f"  oneDNN backend: {torch.backends.mkldnn.is_available()}")
        return True
    except Exception as e:
        print(f"{check_mark(False)} Native XPU features: Error - {e}")
        return False

def test_heartlib():
    """Test HeartLib installation"""
    try:
        from heartlib.heartmula.modeling_heartmula import HeartMuLa
        from heartlib.heartcodec.modeling_heartcodec import HeartCodec
        print(f"{check_mark(True)} HeartLib: Installed")
        return True
    except ImportError as e:
        print(f"{check_mark(False)} HeartLib: Not installed - {e}")
        print("  Install with: pip install -e .")
        return False

def test_checkpoints(ckpt_path="./ckpt"):
    """Test checkpoint availability"""
    mula_path = os.path.join(ckpt_path, "HeartMuLa-oss-3B")
    codec_path = os.path.join(ckpt_path, "HeartCodec-oss")
    
    mula_exists = os.path.exists(mula_path)
    codec_exists = os.path.exists(codec_path)
    
    print(f"{check_mark(mula_exists)} HeartMuLa checkpoint: {mula_path}")
    print(f"{check_mark(codec_exists)} HeartCodec checkpoint: {codec_path}")
    
    if not (mula_exists and codec_exists):
        print("  Download with: python download_models.py")
    
    return mula_exists and codec_exists

def test_environment_vars():
    """Test optimization environment variables"""
    vars_to_check = [
        "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
        "SYCL_CACHE_PERSISTENT",
        "IPEX_XPU_ONEDNN_LAYOUT_OPT",
    ]
    
    all_set = True
    for var in vars_to_check:
        is_set = os.environ.get(var) is not None
        print(f"{check_mark(is_set)} {var}: {os.environ.get(var, 'Not set')}")
        all_set = all_set and is_set
    
    if not all_set:
        print("\n  Set with: source ./run_optimized.sh")
    
    return all_set

def test_xpu_features():
    """Test XPU-specific features"""
    try:
        import torch
        if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
            return False
        
        # Test basic tensor operations
        x = torch.randn(100, 100, dtype=torch.bfloat16).to('xpu')
        y = torch.matmul(x, x.t())
        torch.xpu.synchronize()
        print(f"{check_mark(True)} XPU tensor operations: Working")
        
        # Test autocast
        with torch.autocast(device_type='xpu', dtype=torch.bfloat16):
            z = torch.matmul(x, y)
        torch.xpu.synchronize()
        print(f"{check_mark(True)} XPU autocast: Working")
        
        return True
    except Exception as e:
        print(f"{check_mark(False)} XPU features: Error - {e}")
        return False

def test_model_loading():
    """Test optimized model loading"""
    try:
        import torch
        from heartlib.heartmula.modeling_heartmula import HeartMuLa
        
        if not os.path.exists("./ckpt/HeartMuLa-oss-3B"):
            print(f"{check_mark(False)} Model loading: Checkpoint not found")
            return False
        
        import time
        start = time.time()
        
        model = HeartMuLa.from_pretrained(
            "./ckpt/HeartMuLa-oss-3B",
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        
        load_time = time.time() - start
        print(f"{check_mark(True)} Model loading: {load_time:.2f}s")
        
        if load_time < 1.5:
            print("  ⚡ Excellent! (< 1.5s)")
        elif load_time < 2.5:
            print("  ✓ Good (< 2.5s)")
        else:
            print("  ⚠ Slow (> 2.5s) - check optimizations")
        
        del model
        return True
    except Exception as e:
        print(f"{check_mark(False)} Model loading: Error - {e}")
        return False

def main():
    print_header("HeartMuLa XPU Optimization Validator")
    
    results = {}
    
    # Test 1: PyTorch XPU
    print_header("1. PyTorch XPU Support")
    results['xpu'], pytorch_version = test_pytorch_xpu()
    if pytorch_version:
        print(f"  PyTorch version: {pytorch_version}")
    
    # Test 2: Native XPU Features
    print_header("2. Native PyTorch XPU Features")
    results['native_xpu'] = test_native_xpu_features()
    
    # Test 3: HeartLib
    print_header("3. HeartLib Installation")
    results['heartlib'] = test_heartlib()
    
    # Test 4: Checkpoints
    print_header("4. Model Checkpoints")
    results['checkpoints'] = test_checkpoints()
    
    # Test 5: Environment Variables
    print_header("5. Optimization Environment Variables")
    results['env_vars'] = test_environment_vars()
    
    # Test 6: XPU Features
    print_header("6. XPU Features")
    results['xpu_features'] = test_xpu_features()
    
    # Test 7: Model Loading (if checkpoints exist)
    if results['checkpoints'] and results['heartlib']:
        print_header("7. Optimized Model Loading Test")
        results['model_loading'] = test_model_loading()
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"Tests passed: {passed}/{total}\n")
    
    for test, result in results.items():
        print(f"  {check_mark(result)} {test.replace('_', ' ').title()}")
    
    print()
    
    if passed == total:
        print("✓ All checks passed! System is ready for optimized inference.")
        print("\nRun with: ./run_optimized.sh --lyrics ./inputs/lyrics/default.txt")
        return 0
    elif results.get('xpu') and results.get('heartlib'):
        print("⚠ Some optional optimizations are missing.")
        print("  The system will work, but performance may not be optimal.")
        print("\nFor best performance:")
        if not results.get('env_vars'):
            print("  - Use: ./run_optimized.sh instead of direct python")
        return 0
    else:
        print("✗ Critical components are missing.")
        print("\nPlease fix the issues above before running HeartMuLa.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

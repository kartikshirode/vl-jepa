#!/usr/bin/env python3
"""Check CUDA availability and diagnose issues."""

import sys
import platform

if platform.system() == "Windows":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("Python:", sys.version)
print("Python executable:", sys.executable)

try:
    import torch
    print("\n=== PyTorch Info ===")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version (compiled):", torch.version.cuda)
    
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("CUDA memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
        
        # Test CUDA operation
        x = torch.randn(100, 100, device='cuda')
        y = torch.matmul(x, x)
        print("CUDA test (matmul): PASSED")
    else:
        print("\n=== Diagnosing CUDA Issue ===")
        print("torch.backends.cudnn.enabled:", torch.backends.cudnn.enabled)
        print("torch.backends.cudnn.version():", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A")
        
        # Check if it's a JetPack/ARM issue
        import platform
        print("\nPlatform:", platform.platform())
        print("Machine:", platform.machine())
        
        # Check CUDA libraries
        import os
        if platform.system() == "Windows":
            cuda_paths = [
                os.environ.get("CUDA_PATH", ""),
                os.environ.get("CUDA_PATH_V12_1", ""),
                os.environ.get("CUDA_PATH_V11_8", ""),
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            ]
            env_var = "PATH"
        else:
            cuda_paths = [
                "/usr/local/cuda",
                "/usr/local/cuda/lib64",
                "/usr/lib/aarch64-linux-gnu",
            ]
            env_var = "LD_LIBRARY_PATH"
        print("\nCUDA library paths:")
        for p in cuda_paths:
            if not p:
                continue
            exists = os.path.exists(p)
            print(f"  {p}: {'EXISTS' if exists else 'NOT FOUND'}")
        print(f"\n{env_var}:", os.environ.get(env_var, "NOT SET"))
        
        # Check if torch was built with CUDA
        print("\ntorch.cuda._is_compiled():", torch.cuda._is_compiled() if hasattr(torch.cuda, '_is_compiled') else "N/A")
        
except ImportError as e:
    print("PyTorch import error:", e)
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""
Comprehensive test to verify SPIN integration is working correctly.
This tests all the critical components needed for inference.
"""

import sys
import os
from pathlib import Path

# CRITICAL: Import chumpy compatibility shim BEFORE any SPIN/SMPL imports
import chumpy_compat  # noqa: F401

def test_imports():
    """Test that all required modules can be imported."""
    print("=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    
    tests = []
    
    # Test standard dependencies
    print("\n1. Testing standard dependencies...")
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ torch: {e}")
        tests.append(False)
    
    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ numpy: {e}")
        tests.append(False)
    
    try:
        import cv2
        print(f"  ✓ opencv {cv2.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ opencv: {e}")
        tests.append(False)
    
    try:
        import smplx
        print(f"  ✓ smplx {smplx.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ smplx: {e}")
        tests.append(False)
    
    # Test SPIN modules
    print("\n2. Testing SPIN module imports...")
    spin_root = Path(__file__).parent / "spin_src"
    if str(spin_root) not in sys.path:
        sys.path.insert(0, str(spin_root))
    
    try:
        from models import hmr
        print("  ✓ SPIN HMR model")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ SPIN HMR: {e}")
        tests.append(False)
        return False
    
    try:
        from models import SMPL
        print("  ✓ SPIN SMPL model")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ SPIN SMPL: {e}")
        tests.append(False)
        return False
    
    try:
        import constants as spin_constants
        print("  ✓ SPIN constants")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ SPIN constants: {e}")
        tests.append(False)
    
    try:
        import config as spin_config
        print("  ✓ SPIN config")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ SPIN config: {e}")
        tests.append(False)
    
    return all(tests)


def test_spin_initialization():
    """Test that SPIN model can be initialized."""
    print("\n" + "=" * 60)
    print("Testing SPIN Model Initialization")
    print("=" * 60)
    
    spin_root = Path(__file__).parent / "spin_src"
    if str(spin_root) not in sys.path:
        sys.path.insert(0, str(spin_root))
    
    # Change to SPIN directory for relative paths in config
    original_cwd = os.getcwd()
    os.chdir(spin_root)
    
    try:
        import torch
        from models import hmr
        import config as spin_config
        
        print("\n1. Loading HMR model...")
        smpl_mean_params = spin_config.SMPL_MEAN_PARAMS
        
        if not os.path.exists(smpl_mean_params):
            print(f"  ✗ SMPL mean params not found: {smpl_mean_params}")
            return False
        
        model = hmr(smpl_mean_params)
        print(f"  ✓ HMR model created")
        
        # Test loading checkpoint
        print("\n2. Checking model checkpoint...")
        checkpoint_path = "data/model_checkpoint.pt"
        
        if not os.path.exists(checkpoint_path):
            print(f"  ✗ Checkpoint not found: {checkpoint_path}")
            return False
        
        print(f"  ✓ Checkpoint exists ({os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB)")
        
        # Test SMPL model
        print("\n3. Initializing SMPL model...")
        from models import SMPL
        smpl_model_dir = spin_config.SMPL_MODEL_DIR
        
        if not os.path.exists(smpl_model_dir):
            print(f"  ✗ SMPL model dir not found: {smpl_model_dir}")
            return False
        
        smpl = SMPL(smpl_model_dir, batch_size=1, create_transl=False)
        print(f"  ✓ SMPL model initialized")
        print(f"  ✓ SMPL faces shape: {smpl.faces.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def test_spin_wrapper():
    """Test our SpinModelWrapper."""
    print("\n" + "=" * 60)
    print("Testing SpinModelWrapper Integration")
    print("=" * 60)
    
    try:
        # Import our project config (not SPIN's config)
        # Need to use absolute path to avoid conflict with SPIN's config
        project_root = str(Path(__file__).parent.absolute())
        config_path = os.path.join(project_root, 'config.py')
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("project_config", config_path)
        project_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(project_config)
        
        from src.reconstruction.spin_wrapper import SpinModelWrapper
        
        print("\n1. Initializing SpinModelWrapper...")
        wrapper = SpinModelWrapper(project_config.SPIN_ROOT)
        print("  ✓ Wrapper initialized successfully")
        print(f"  ✓ Device: {wrapper.device}")
        print(f"  ✓ Model loaded: {wrapper.model is not None}")
        print(f"  ✓ SMPL loaded: {wrapper.smpl is not None}")
        print(f"  ✓ Faces shape: {wrapper.faces.shape if wrapper.faces is not None else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("SPIN Integration Test Suite")
    print("=" * 60)
    print()
    
    # Run tests
    results = []
    
    print("\n" + "▶" * 20)
    print("TEST 1/3: Module Imports")
    print("▶" * 20)
    results.append(("Module Imports", test_imports()))
    
    print("\n" + "▶" * 20)
    print("TEST 2/3: SPIN Initialization")
    print("▶" * 20)
    results.append(("SPIN Initialization", test_spin_initialization()))
    
    print("\n" + "▶" * 20)
    print("TEST 3/3: SpinModelWrapper")
    print("▶" * 20)
    results.append(("SpinModelWrapper", test_spin_wrapper()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour SPIN integration is working correctly.")
        print("You can now run the Streamlit app:")
        print("  streamlit run app.py")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease review the errors above.")
        print("See COMPATIBILITY_FIXES.md for troubleshooting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Final validation script - Verifies the complete cricket bowling analysis pipeline.
Run this to confirm everything is working correctly.
"""

import chumpy_compat  # MUST be first

import sys
import os
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_critical_imports():
    """Test all critical imports work."""
    print_header("Testing Critical Imports")
    
    errors = []
    
    # Test standard dependencies
    try:
        import torch
        print(f"✅ torch {torch.__version__}")
    except Exception as e:
        errors.append(f"torch: {e}")
        print(f"❌ torch: {e}")
    
    try:
        import streamlit
        print(f"✅ streamlit {streamlit.__version__}")
    except Exception as e:
        errors.append(f"streamlit: {e}")
        print(f"❌ streamlit: {e}")
    
    try:
        import mediapipe
        print(f"✅ mediapipe {mediapipe.__version__}")
    except Exception as e:
        errors.append(f"mediapipe: {e}")
        print(f"❌ mediapipe: {e}")
    
    try:
        import cv2
        print(f"✅ opencv {cv2.__version__}")
    except Exception as e:
        errors.append(f"opencv: {e}")
        print(f"❌ opencv: {e}")
    
    # Test SPIN integration
    try:
        from src.reconstruction.spin_wrapper import SpinModelWrapper
        print(f"✅ SpinModelWrapper import successful")
    except Exception as e:
        errors.append(f"SpinModelWrapper: {e}")
        print(f"❌ SpinModelWrapper: {e}")
    
    # Test chumpy compatibility
    try:
        import chumpy
        print(f"✅ chumpy loaded (with compatibility patches)")
    except Exception as e:
        errors.append(f"chumpy: {e}")
        print(f"❌ chumpy: {e}")
    
    return len(errors) == 0

def test_spin_initialization():
    """Test that SPIN model can be fully initialized."""
    print_header("Testing SPIN Model Initialization")
    
    try:
        from src.reconstruction.spin_wrapper import SpinModelWrapper
        from app_config import SPIN_ROOT
        
        print("Initializing SpinModelWrapper (this may take a minute)...")
        wrapper = SpinModelWrapper(SPIN_ROOT)
        
        print(f"✅ Wrapper initialized successfully")
        print(f"✅ Device: {wrapper.device}")
        print(f"✅ Model loaded: {wrapper.model is not None}")
        print(f"✅ SMPL loaded: {wrapper.smpl is not None}")
        print(f"✅ Faces shape: {wrapper.faces.shape}")
        print(f"✅ Input resolution: {wrapper.input_res}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_project_modules():
    """Test all project modules can be imported."""
    print_header("Testing Project Modules")
    
    modules = [
        ("config", "Project configuration"),
        ("src.ingestion.video_utils", "Video processing"),
        ("src.pose2d.mediapipe_runner", "MediaPipe Pose"),
        ("src.delivery.delivery_detector", "Delivery detection"),
        ("src.reconstruction.spin_wrapper", "SPIN wrapper"),
        ("src.analysis.metrics", "Metrics computation"),
        ("src.visualization.render_utils", "Visualization"),
    ]
    
    errors = []
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✅ {description:25s} ({module_name})")
        except Exception as e:
            errors.append(f"{description}: {e}")
            print(f"❌ {description:25s} ({module_name}): {e}")
    
    return len(errors) == 0

def test_data_files():
    """Test that all required data files are present."""
    print_header("Testing Required Data Files")
    
    spin_root = Path(__file__).parent / "spin_src"
    
    files_to_check = [
        ("SPIN checkpoint", spin_root / "data" / "model_checkpoint.pt"),
        ("SMPL neutral", spin_root / "data" / "smpl" / "SMPL_NEUTRAL.pkl"),
        ("SMPL male", spin_root / "data" / "smpl" / "SMPL_MALE.pkl"),
        ("SMPL female", spin_root / "data" / "smpl" / "SMPL_FEMALE.pkl"),
        ("J_regressor_extra", spin_root / "data" / "J_regressor_extra.npy"),
        ("J_regressor_h36m", spin_root / "data" / "J_regressor_h36m.npy"),
        ("SMPL mean params", spin_root / "data" / "smpl_mean_params.npz"),
    ]
    
    errors = []
    total_size = 0
    
    for description, filepath in files_to_check:
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            total_size += size_mb
            print(f"✅ {description:20s} ({size_mb:6.1f} MB)")
        else:
            errors.append(f"{description}: {filepath}")
            print(f"❌ {description:20s} NOT FOUND: {filepath}")
    
    print(f"\nTotal data size: {total_size:.1f} MB")
    return len(errors) == 0

def test_compatibility_patches():
    """Test that compatibility patches are applied."""
    print_header("Testing Compatibility Patches")
    
    errors = []
    
    # Test inspect.getargspec patch
    import inspect
    if hasattr(inspect, 'getargspec'):
        print("✅ inspect.getargspec patched (points to getfullargspec)")
    else:
        errors.append("inspect.getargspec not patched")
        print("❌ inspect.getargspec not patched")
    
    # Test numpy type alias patches
    import numpy as np
    aliases_to_check = ['bool', 'int', 'float', 'complex', 'object', 'unicode', 'str']
    for alias in aliases_to_check:
        if hasattr(np, alias):
            print(f"✅ numpy.{alias} alias present")
        else:
            errors.append(f"numpy.{alias} not patched")
            print(f"❌ numpy.{alias} alias not present")
    
    return len(errors) == 0

def main():
    """Run all validation tests."""
    print("\n" + "▓" * 70)
    print("▓" + " " * 68 + "▓")
    print("▓" + "  CRICKET BOWLING 3D ANALYSIS - FINAL VALIDATION".center(68) + "▓")
    print("▓" + " " * 68 + "▓")
    print("▓" * 70)
    
    results = []
    
    # Run all tests
    results.append(("Compatibility Patches", test_compatibility_patches()))
    results.append(("Critical Imports", test_critical_imports()))
    results.append(("Project Modules", test_project_modules()))
    results.append(("Data Files", test_data_files()))
    results.append(("SPIN Initialization", test_spin_initialization()))
    
    # Print summary
    print_header("VALIDATION SUMMARY")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8s} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "▓" * 70)
    if all_passed:
        print("▓" + " " * 68 + "▓")
        print("▓" + "  🎉 ALL VALIDATION TESTS PASSED! 🎉".center(68) + "▓")
        print("▓" + " " * 68 + "▓")
        print("▓" + "  Your app is ready to use!".center(68) + "▓")
        print("▓" + "  Run: python -m streamlit run app.py".center(68) + "▓")
        print("▓" + " " * 68 + "▓")
        print("▓" * 70)
        return 0
    else:
        print("▓" + " " * 68 + "▓")
        print("▓" + "  ❌ SOME TESTS FAILED".center(68) + "▓")
        print("▓" + " " * 68 + "▓")
        print("▓" + "  Please review the errors above.".center(68) + "▓")
        print("▓" + "  See COMPATIBILITY_FIXES.md for troubleshooting.".center(68) + "▓")
        print("▓" + " " * 68 + "▓")
        print("▓" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())

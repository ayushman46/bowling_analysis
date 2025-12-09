#!/usr/bin/env python3
"""
Validation script to check if the SPIN cricket bowling analysis app is properly set up.
Run this before using the main Streamlit app to ensure all dependencies are installed
and required data files are in place.
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_mark(condition):
    return f"{GREEN}✓{RESET}" if condition else f"{RED}✗{RESET}"

def print_header(text):
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}\n")

def check_python_version():
    """Check Python version is 3.8+"""
    version = sys.version_info
    is_ok = version.major == 3 and version.minor >= 8
    print(f"{check_mark(is_ok)} Python version: {version.major}.{version.minor}.{version.micro}")
    if not is_ok:
        print(f"  {RED}ERROR: Python 3.8+ required{RESET}")
    return is_ok

def check_module(module_name, package_name=None):
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        print(f"{check_mark(True)} {package_name or module_name}")
        return True
    except ImportError:
        print(f"{check_mark(False)} {package_name or module_name}")
        print(f"  {YELLOW}Install with: pip install {package_name or module_name}{RESET}")
        return False

def check_file(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    print(f"{check_mark(exists)} {description}")
    if exists:
        size = os.path.getsize(filepath)
        print(f"  Location: {filepath}")
        print(f"  Size: {size / 1024 / 1024:.1f} MB")
    else:
        print(f"  {YELLOW}Missing: {filepath}{RESET}")
    return exists

def check_directory(dirpath, description):
    """Check if a directory exists"""
    exists = os.path.isdir(dirpath)
    print(f"{check_mark(exists)} {description}")
    if exists:
        print(f"  Location: {dirpath}")
    else:
        print(f"  {YELLOW}Missing: {dirpath}{RESET}")
    return exists

def main():
    print_header("Cricket Bowling 3D Analysis - Setup Validation")
    
    project_root = Path(__file__).parent
    spin_root = project_root / "spin_src"
    
    all_checks = []
    
    # Check Python version
    print_header("1. Python Version")
    all_checks.append(check_python_version())
    
    # Check main dependencies
    print_header("2. Main Python Dependencies")
    dependencies = [
        ("streamlit", "streamlit"),
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("PIL", "Pillow"),
        ("trimesh", "trimesh"),
        ("pyrender", "pyrender"),
    ]
    
    for module, package in dependencies:
        all_checks.append(check_module(module, package))
    
    # Check SPIN-specific dependencies
    print_header("3. SPIN Dependencies")
    spin_deps = [
        ("smplx", "smplx"),
        ("scipy", "scipy"),
        ("skimage", "scikit-image"),
    ]
    
    for module, package in spin_deps:
        all_checks.append(check_module(module, package))
    
    # Check SPIN repository structure
    print_header("4. SPIN Repository Structure")
    all_checks.append(check_directory(spin_root, "SPIN repository (spin_src/)"))
    
    if spin_root.exists():
        spin_files = [
            (spin_root / "models" / "hmr.py", "HMR model"),
            (spin_root / "models" / "smpl.py", "SMPL model"),
            (spin_root / "constants.py", "Constants"),
            (spin_root / "config.py", "Config"),
        ]
        
        for filepath, desc in spin_files:
            all_checks.append(check_file(filepath, desc))
    
    # Check SPIN data files
    print_header("5. SPIN Model Data Files")
    data_dir = spin_root / "data"
    
    all_checks.append(check_directory(data_dir, "Data directory"))
    
    if data_dir.exists():
        data_files = [
            (data_dir / "model_checkpoint.pt", "SPIN checkpoint (~300MB)"),
            (data_dir / "smpl_mean_params.npz", "SMPL mean parameters"),
        ]
        
        for filepath, desc in data_files:
            result = check_file(filepath, desc)
            all_checks.append(result)
            if not result:
                print(f"  {YELLOW}Run: ./setup_spin.sh to download{RESET}")
    
    # Check SMPL model files
    print_header("6. SMPL Body Model Files")
    smpl_dir = data_dir / "smpl"
    
    all_checks.append(check_directory(smpl_dir, "SMPL models directory"))
    
    if smpl_dir.exists():
        smpl_models = [
            (smpl_dir / "SMPL_NEUTRAL.pkl", "SMPL Neutral model"),
            (smpl_dir / "SMPL_MALE.pkl", "SMPL Male model (optional)"),
            (smpl_dir / "SMPL_FEMALE.pkl", "SMPL Female model (optional)"),
        ]
        
        for filepath, desc in smpl_models:
            result = check_file(filepath, desc)
            if "Neutral" in desc:
                all_checks.append(result)
            
            if not result and "Neutral" in desc:
                print(f"  {YELLOW}Download from: http://smplify.is.tue.mpg.de{RESET}")
            elif not result:
                print(f"  {YELLOW}Download from: http://smpl.is.tue.mpg.de{RESET}")
    
    # Check project structure
    print_header("7. Project Structure")
    project_files = [
        (project_root / "app.py", "Main Streamlit app"),
        (project_root / "config.py", "Configuration"),
        (project_root / "src" / "ingestion" / "video_utils.py", "Video ingestion"),
        (project_root / "src" / "pose2d" / "mediapipe_runner.py", "2D pose estimation"),
        (project_root / "src" / "delivery" / "delivery_detector.py", "Delivery detection"),
        (project_root / "src" / "reconstruction" / "spin_wrapper.py", "SPIN wrapper"),
        (project_root / "src" / "analysis" / "metrics.py", "Metrics computation"),
        (project_root / "src" / "visualization" / "render_utils.py", "Visualization"),
    ]
    
    for filepath, desc in project_files:
        all_checks.append(check_file(filepath, desc))
    
    # Summary
    print_header("Summary")
    
    total = len(all_checks)
    passed = sum(all_checks)
    failed = total - passed
    
    print(f"Total checks: {total}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    
    if failed == 0:
        print(f"\n{GREEN}{'=' * 60}{RESET}")
        print(f"{GREEN}All checks passed! You're ready to use the app.{RESET}")
        print(f"{GREEN}{'=' * 60}{RESET}\n")
        print("To start the app, run:")
        print(f"  {BLUE}streamlit run app.py{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}{'=' * 60}{RESET}")
        print(f"{YELLOW}Some checks failed. Please review the issues above.{RESET}")
        print(f"{YELLOW}{'=' * 60}{RESET}\n")
        
        print("Common fixes:")
        print(f"1. Install dependencies:")
        print(f"   {BLUE}pip install -r requirements.txt{RESET}")
        print(f"   {BLUE}pip install -r spin_src/requirements.txt{RESET}")
        print(f"\n2. Download SPIN data:")
        print(f"   {BLUE}./setup_spin.sh{RESET}")
        print(f"\n3. Get SMPL models from official sources (see README.md)")
        return 1

if __name__ == "__main__":
    sys.exit(main())

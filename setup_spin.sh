Error during processing: Could not import SPIN modules from /Users/ayush/Downloads/bowlin/spin_src. Make sure spin_src is the official SPIN repo. Error: cannot import name 'ModelOutput' from 'smplx.body_models' (/Users/ayush/Downloads/bowlin/.venv/lib/python3.12/site-packages/smplx/body_models.py)#!/bin/bash
# Setup script to download SPIN model data files

set -e  # Exit on error

echo "=== SPIN Data Setup Script ==="
echo ""
echo "This script will download the required SPIN model checkpoint and data files."
echo "It will download approximately 500MB of data."
echo ""

# Navigate to spin_src directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SPIN_DIR="$SCRIPT_DIR/spin_src"

if [ ! -d "$SPIN_DIR" ]; then
    echo "ERROR: spin_src directory not found at $SPIN_DIR"
    echo "Please ensure you have cloned the SPIN repository into spin_src/"
    exit 1
fi

cd "$SPIN_DIR"
echo "Working in: $SPIN_DIR"
echo ""

# Create data directory if it doesn't exist
mkdir -p data
cd data

# Download model checkpoint
if [ ! -f "model_checkpoint.pt" ]; then
    echo "Downloading SPIN model checkpoint..."
    curl -L -o model_checkpoint.pt http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt
    echo "✓ Model checkpoint downloaded"
else
    echo "✓ Model checkpoint already exists"
fi

# Download SMPL mean parameters
if [ ! -f "smpl_mean_params.npz" ]; then
    echo "Downloading SMPL mean parameters..."
    # This is part of the data.tar.gz package
    cd ..
    if [ ! -f "data.tar.gz" ]; then
        curl -L -o data.tar.gz http://visiondata.cis.upenn.edu/spin/data.tar.gz
        tar -xzf data.tar.gz
        rm data.tar.gz
        echo "✓ SPIN data files extracted"
    fi
    cd data
else
    echo "✓ SMPL mean parameters already exist"
fi

# Check for SMPL models
echo ""
echo "Checking SMPL model files..."
SMPL_DIR="smpl"

if [ ! -f "$SMPL_DIR/SMPL_NEUTRAL.pkl" ] || [ ! -f "$SMPL_DIR/SMPL_MALE.pkl" ] || [ ! -f "$SMPL_DIR/SMPL_FEMALE.pkl" ]; then
    echo ""
    echo "WARNING: SMPL model files not found in $SPIN_DIR/data/smpl/"
    echo ""
    echo "You need to manually download the SMPL models from:"
    echo "  - Neutral model: http://smplify.is.tue.mpg.de"
    echo "  - Male/Female models: http://smpl.is.tue.mpg.de"
    echo ""
    echo "After downloading, rename them to:"
    echo "  - SMPL_NEUTRAL.pkl"
    echo "  - SMPL_MALE.pkl"
    echo "  - SMPL_FEMALE.pkl"
    echo ""
    echo "And place them in: $SPIN_DIR/data/smpl/"
    echo ""
    echo "NOTE: The user has indicated they already have these files."
    echo "Please verify they are in the correct location."
else
    echo "✓ SMPL model files found:"
    ls -lh $SMPL_DIR/*.pkl
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Ensure SMPL models are in place (see warning above if any)"
echo "2. Install Python dependencies:"
echo "   pip install -r requirements.txt"
echo "   pip install -r spin_src/requirements.txt"
echo "3. Run the app:"
echo "   streamlit run app.py"
echo ""

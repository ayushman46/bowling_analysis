#!/bin/bash
#
# Quick start script for Cricket Bowling 3D Analysis App
#
# This script starts the Streamlit application with the correct environment.
#

set -e  # Exit on error

echo "=================================================================================================="
echo "🏏 Cricket Bowling 3D Analysis App - Starting..."
echo "=================================================================================================="
echo ""

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not active. Activating..."
    if [ ! -d ".venv" ]; then
        echo "❌ Error: .venv directory not found. Please run setup first."
        exit 1
    fi
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "✅ Virtual environment already active: $VIRTUAL_ENV"
fi

echo ""
echo "Starting Streamlit application..."
echo ""
echo "▸▸▸ The app will open in your browser at: http://localhost:8501"
echo "▸▸▸ Press Ctrl+C to stop the application"
echo ""

# Start streamlit
python -m streamlit run app.py


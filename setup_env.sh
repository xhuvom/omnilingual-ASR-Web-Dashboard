#!/bin/bash

# Exit on error
set -e

VENV_NAME="asr_venv"

echo "================================================"
echo "Setting up Bangla ASR Fine-Tuning Environment"
echo "================================================"

# 1. Create Virtual Environment
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: $VENV_NAME..."
    python3 -m venv "$VENV_NAME"
else
    echo "Virtual environment $VENV_NAME already exists."
fi

# 2. Upgrade Pip
echo "Upgrading pip..."
./$VENV_NAME/bin/python -m pip install --upgrade pip

# 3. Install PyTorch and core dependencies
echo "Installing PyTorch and core ML libraries..."
./$VENV_NAME/bin/python -m pip install torch==2.5.1 torchaudio==2.5.1

# 4. Install omnilingual-asr package (local installation)
if [ -f "pyproject.toml" ]; then
    echo "Installing omnilingual-asr package from local directory..."
    ./$VENV_NAME/bin/python -m pip install -e .
else
    echo "Warning: pyproject.toml not found. Skipping omnilingual-asr installation."
fi

# 5. Install remaining dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing remaining requirements from requirements.txt..."
    ./$VENV_NAME/bin/python -m pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo "Environment: $VENV_NAME"
echo ""
echo "Installed packages:"
echo "  - PyTorch 2.5.1"
echo "  - omnilingual-asr (local)"
echo "  - transformers, peft, datasets"
echo "  - librosa, soundfile (audio)"
echo "  - All training/inference dependencies"
echo ""
echo "To activate the environment:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To run training:"
echo "  ./run_training.sh"
echo ""
echo "To run inference:"
echo "  ./$VENV_NAME/bin/python inference_asr_lora.py --audio_path <path_to_audio>"
echo "================================================"

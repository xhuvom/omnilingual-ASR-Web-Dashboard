#!/bin/bash

# Training script for Bangla ASR CTC-1B model with LoRA on A40 GPU

echo "==================================="
echo "Training Bangla ASR on A40 GPU"
echo "==================================="
echo "Start time: $(date)"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

echo "Starting training..."
echo ""

if ./asr_venv/bin/python train_asr_lora.py; then
    echo ""
    echo "==================================="
    echo "Training completed successfully!"
    echo "End time: $(date)"
    echo "==================================="
else
    echo ""
    echo "Training failed!"
    exit 1
fi

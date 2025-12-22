
import datasets
from datasets import Audio
import soundfile as sf
import torchaudio
import torch
import sys

print(f"Datasets version: {datasets.__version__}")
print(f"SoundFile version: {sf.__version__}")
print(f"TorchAudio version: {torchaudio.__version__}")

try:
    print("Testing SoundFile import...")
    import soundfile
    print("SoundFile imported successfully.")
except ImportError as e:
    print(f"SoundFile import failed: {e}")

try:
    print("Testing TorchAudio backend...")
    print(torchaudio.list_audio_backends())
except Exception as e:
    print(f"TorchAudio backend check failed: {e}")

print("Attempting to load a dummy audio feature...")
try:
    # Create a dummy audio file
    import numpy as np
    dummy_audio = np.random.uniform(-1, 1, 16000)
    sf.write('test_audio.wav', dummy_audio, 16000)
    
    # Try using datasets.Audio
    audio_feature = Audio(sampling_rate=16000)
    decoded = audio_feature.decode_example({"path": "test_audio.wav", "bytes": None})
    print("Datasets Audio decoding successful!")
    print(f"Shape: {decoded['array'].shape}")
except Exception as e:
    print(f"Datasets Audio decoding failed: {e}")
    # Print full traceback
    import traceback
    traceback.print_exc()

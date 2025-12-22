import torch
import librosa
import os
import argparse
from transformers import Wav2Vec2ForCTC, AutoProcessor
from peft import PeftModel, PeftConfig

def main():
    parser = argparse.ArgumentParser(description="Evaluate Inference for Bangla ASR LoRA Model")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file (.wav)")
    parser.add_argument("--model_path", type=str, default="mms-1b-bangla-lora/final_model", help="Path to the trained LoRA adapter")
    parser.add_argument("--base_model", type=str, default="facebook/mms-1b-all", help="Base model ID")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Processor
    print(f"Loading processor from {args.base_model} with target_lang='ben'...")
    processor = AutoProcessor.from_pretrained(args.base_model, target_lang="ben")
    
    # 2. Load Base Model
    print(f"Loading base model: {args.base_model}...")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ignore_mismatched_sizes=True,
        target_lang="ben",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # 3. Load LoRA Adapter
    print(f"Loading LoRA adapter from {args.model_path}...")
    model = PeftModel.from_pretrained(model, args.model_path)
    model.to(device)
    model.eval()

    # 4. Load Audio
    print(f"Loading audio: {args.audio_path}...")
    # Resample to 16kHz
    audio_input, sr = librosa.load(args.audio_path, sr=16000)

    # 5. Preprocess
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if device == "cuda":
        inputs["input_values"] = inputs["input_values"].half()

    # 6. Inference
    print("Running inference...")
    with torch.no_grad():
        logits = model(**inputs).logits

    # 7. Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    print("-" * 40)
    print("TRANSCRIPTION:")
    print("-" * 40)
    print(transcription)
    print("-" * 40)

if __name__ == "__main__":
    main()

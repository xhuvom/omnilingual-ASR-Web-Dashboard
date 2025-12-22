import os
import argparse
import pandas as pd
import glob
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset CSV for ASR training")
    parser.add_argument("--audio_dir", required=True, help="Path to directory containing audio files (.wav, .mp3, .flac)")
    parser.add_argument("--output_csv", default="generated_dataset.csv", help="Output CSV filename")
    parser.add_argument("--transcripts", help="Optional: Path to a text file containing transcripts (line by line matching audios, or format: filename|transcript)")
    
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"Error: Directory {audio_dir} does not exist.")
        return

    # Scan for audio files
    audio_extensions = ["*.wav", "*.mp3", "*.flac"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(audio_dir.glob(ext)) + list(audio_dir.glob("**/" + ext))) # Recursive search
    
    if not audio_files:
        print("No audio files found!")
        return
        
    print(f"Found {len(audio_files)} audio files.")
    
    data = []
    
    # Check if transcripts provided
    transcripts_map = {}
    if args.transcripts:
        print(f"Loading transcripts from {args.transcripts}...")
        with open(args.transcripts, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    # Format: filename|text
                    parts = line.split("|", 1)
                    transcripts_map[parts[0].strip()] = parts[1].strip()
                else:
                    # Generic fallback or needs custom logic
                    pass
    
    for audio_path in audio_files:
        # Relative path from audio_dir
        rel_path = audio_path.relative_to(audio_dir)
        filename = audio_path.name
        
        # Try to find transcript
        sentence = ""
        if filename in transcripts_map:
            sentence = transcripts_map[filename]
        elif audio_path.stem in transcripts_map:
            sentence = transcripts_map[audio_path.stem]
            
        data.append({
            "audio_path": str(rel_path),
            "sentence": sentence
        })
    
    df = pd.DataFrame(data)
    output_path = audio_dir / args.output_csv
    df.to_csv(output_path, index=False)
    
    print(f"Dataset CSV created at: {output_path}")
    print("Columns: ", df.columns.tolist())
    print("Example rows:")
    print(df.head())
    print("-" * 30)
    print("Instructions:")
    print(1, f"If 'sentence' column is empty, please fill it with transcripts manually or provide a formatting transcript file.")
    print(2, f"Use this directory path in your training script.")

if __name__ == "__main__":
    main()

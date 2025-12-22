import os
import argparse
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
import time

def setup_directories(output_dir):
    """Creates output directory and audio subdirectory."""
    output_path = Path(output_dir)
    audio_path = output_path / "audios"
    
    if not output_path.exists():
        print(f"Creating output directory: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        
    if not audio_path.exists():
        print(f"Creating audio subdirectory: {audio_path}")
        audio_path.mkdir(exist_ok=True)
        
    return output_path, audio_path

def process_bangla_quantity(source_dir, output_audio_dir, limit=None):
    """
    Process 'bangla_voice_quantity' dataset.
    - source_dir: Root of bangla_voice_quantity
    - output_audio_dir: Destination for audio files
    - limit: Max files to process (for testing)
    """
    print(f"\n--- Processing Bangla Quantity Dataset ---")
    source_path = Path(source_dir)
    csv_path = source_path / "transcriptions.csv"
    speech_dir = source_path / "speech"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return []
        
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} metadata entries.")
    
    # Map filename to full path
    print("Scanning speech directory for files...")
    file_map = {}
    # Recursive search for all wav files in speech dir
    all_files = list(speech_dir.rglob("*.wav"))
    print(f"Found {len(all_files)} audio files on disk.")
    
    for f in all_files:
        file_map[f.name] = f
        
    dataset_entries = []
    
    # Process rows
    total = len(df)
    if limit:
        total = min(total, limit)
        df = df.head(total)
        print(f"Limiting to {total} files as requested.")

    success_count = 0
    missing_count = 0
    
    print("Copying files...")
    # Using tqdm for progress tracking
    for idx, row in tqdm(df.iterrows(), total=total, unit="file"):
        filename = row['File Name']
        transcript = row['Bengali Transcription']
        
        # Determine source file
        if filename in file_map:
            src_file = file_map[filename]
        else:
            # Try appending .wav if missing
            if not str(filename).endswith('.wav') and (str(filename) + '.wav') in file_map:
                src_file = file_map[str(filename) + '.wav']
            else:
                missing_count += 1
                continue
        
        # Create new unique filename
        new_filename = f"bq_{src_file.name}"
        dst_file = output_audio_dir / new_filename
        
        # Copy file
        try:
            shutil.copy2(src_file, dst_file)
            dataset_entries.append({
                "audio_path": f"audios/{new_filename}",
                "sentence": transcript,
                "dataset_source": "bangla_quantity"
            })
            success_count += 1
        except Exception as e:
            print(f"Error copying {src_file}: {e}")
            
    print(f"Bangla Quantity: Processed {success_count} files ({missing_count} missing).")
    return dataset_entries

def process_kaggle_bangla(source_dir, output_audio_dir, limit=None):
    """
    Process 'kaggle_bangla' dataset.
    - source_dir: Root of kaggle_bangla
    - output_audio_dir: Destination for audio files
    - limit: Max files to process (for testing)
    """
    print(f"\n--- Processing Kaggle Bangla Dataset ---")
    source_path = Path(source_dir)
    csv_path = source_path / "train.csv"
    mp3_dir = source_path / "train_mp3s"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return []
        
    print(f"Loading metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} metadata entries.")
    
    dataset_entries = []
    
    total = len(df)
    if limit:
        total = min(total, limit)
        df = df.head(total)
        print(f"Limiting to {total} files as requested.")
        
    success_count = 0
    missing_count = 0
    
    print("Copying files...")
    for idx, row in tqdm(df.iterrows(), total=total, unit="file"):
        file_id = row['id']
        sentence = row['sentence']
        
        # Construct filename. Assuming mp3 based on user description and folder name
        filename = f"{file_id}.mp3" 
        src_file = mp3_dir / filename
        
        if not src_file.exists():
            missing_count += 1
            continue
            
        # Create new unique filename
        new_filename = f"kb_{file_id}.mp3"
        dst_file = output_audio_dir / new_filename
        
        # Copy file
        try:
            shutil.copy2(src_file, dst_file)
            dataset_entries.append({
                "audio_path": f"audios/{new_filename}",
                "sentence": sentence,
                "dataset_source": "kaggle_bangla"
            })
            success_count += 1
        except Exception as e:
            print(f"Error copying {src_file}: {e}")

    print(f"Kaggle Bangla: Processed {success_count} files ({missing_count} missing).")
    return dataset_entries

def main():
    parser = argparse.ArgumentParser(description="Combine and prepare Bangla ASR datasets.")
    parser.add_argument("--bq_dir", default="/mnt/sdc1/Bangla_ASR/bangla_voice_quantity", help="Path to Bangla Quantity dataset")
    parser.add_argument("--kb_dir", default="/mnt/sdc1/Bangla_ASR/kaggle_bangla", help="Path to Kaggle Bangla dataset")
    parser.add_argument("--output_dir", default="/mnt/sdc1/Bangla_ASR/combined_dataset", help="Path to updated dataset directory")
    parser.add_argument("--limit", type=int, help="Limit number of files per dataset for testing")
    
    args = parser.parse_args()
    
    print("Starting Dataset Preparation...")
    output_path, audio_path = setup_directories(args.output_dir)
    
    all_data = []
    
    # Process Dataset 1
    bq_data = process_bangla_quantity(args.bq_dir, audio_path, args.limit)
    all_data.extend(bq_data)
    
    # Process Dataset 2
    # Note: User mentioned kaggle_bangla uses 'train_mp3s' as folder
    kb_data = process_kaggle_bangla(args.kb_dir, audio_path, args.limit)
    all_data.extend(kb_data)
    
    print(f"\n--- Finalizing ---")
    if not all_data:
        print("No data collected! Check paths and logic.")
        return
        
    print(f"Total Combined Samples: {len(all_data)}")
    
    # Save CSV
    df = pd.DataFrame(all_data)
    output_csv = output_path / "dataset.csv"
    print(f"Saving metadata to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    print("\nComplete! Dataset is ready for training.")
    print(f"Dataset Path: {output_csv}")
    print(f"Audio Path: {audio_path}")
    print("\nSample rows:")
    print(df.head())

if __name__ == "__main__":
    main()

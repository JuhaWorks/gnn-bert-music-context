import os
import glob
import json
import torch
from audio_features import load_config, process_audio_file
from tqdm import tqdm

def main():
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    
    # Load splits to know which tracks are in our subset
    with open('data/splits/train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)[:560]
    with open('data/splits/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)[:140]
        
    subset_track_ids = set([item["track_id"] for item in train_data + val_data])
    
    audio_files = glob.glob(os.path.join(raw_dir, "**", "*.mp3"), recursive=True)
    filtered_files = []
    for f in audio_files:
        tid = os.path.splitext(os.path.basename(f))[0]
        if tid in subset_track_ids:
            filtered_files.append(f)
            
    print(f"Extracting mel-spectrograms for {len(filtered_files)} subset tracks...")
    
    for filepath in tqdm(filtered_files):
        track_id = os.path.splitext(os.path.basename(filepath))[0]
        mel_path = os.path.join(processed_dir, f"{track_id}_melspec.pt")
        
        if os.path.exists(mel_path):
            continue
            
        res = process_audio_file(filepath, config)
        if res is not None:
            _, log_mel = res
            torch.save(log_mel, mel_path)

if __name__ == "__main__":
    main()

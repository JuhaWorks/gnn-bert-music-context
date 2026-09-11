import os
import glob
import yaml
import librosa
import numpy as np
import torch
from tqdm import tqdm

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def process_audio_file(filepath, config):
    sr = config["audio"]["sample_rate"]
    window_sec = config["audio"]["window_size_sec"]
    
    # Load audio, downsample it, and convert to mono
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

    # Split into fixed windows
    window_samples = int(window_sec * sr)
    num_windows = len(y) // window_samples
    
    if num_windows == 0:
        return None # Too short
        
    windows = []
    for i in range(num_windows):
        start = i * window_samples
        end = start + window_samples
        y_win = y[start:end]
        
        # Extract chroma features (12 bins)
        chroma = librosa.feature.chroma_cqt(y=y_win, sr=sr, n_chroma=config["audio"]["n_chroma"])
        # Average over time to get a single 12-dim vector per window
        chroma_mean = np.mean(chroma, axis=1)
        windows.append(chroma_mean)
        
    # Also extract log-mel spectrogram for CNN baseline
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=config["audio"]["n_mels"])
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # We will return a tuple: (chroma_tensor, log_mel_tensor)
    return torch.tensor(np.array(windows), dtype=torch.float32), torch.tensor(log_mel_spec, dtype=torch.float32)

def main():
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    processed_dir = config["data"]["processed_dir"]
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Assuming FMA-small structure: data/raw/fma_small/000/000002.mp3
    audio_files = glob.glob(os.path.join(raw_dir, "**", "*.mp3"), recursive=True)
    print(f"Found {len(audio_files)} audio files. Starting preprocessing...")
    
    for filepath in tqdm(audio_files):
        track_id = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(processed_dir, f"{track_id}_features.pt")
        
        if os.path.exists(out_path):
            continue
            
        features, log_mel = process_audio_file(filepath, config)
        if features is not None:
            torch.save(features, out_path)
            mel_path = os.path.join(processed_dir, f"{track_id}_melspec.pt")
            torch.save(log_mel, mel_path)

if __name__ == "__main__":
    main()

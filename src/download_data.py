import os
import requests
import zipfile
import subprocess
import pandas as pd
from tqdm import tqdm
import imageio_ffmpeg

FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
MUSICCAPS_CSV_URL = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"

RAW_DIR = "data/raw"

def download_file_resumable(url, output_path):
    """
    Downloads a file with resume support using the Range header.
    This prevents losing progress if the download is interrupted!
    """
    headers = {}
    mode = 'wb'
    initial_pos = 0
    
    # Get the total file size from the server first
    try:
        head_response = requests.head(url, allow_redirects=True)
        total_size = int(head_response.headers.get('content-length', 0))
    except Exception as e:
        print(f"Warning: Could not get file size for {url}. {e}")
        total_size = 0
        
    if os.path.exists(output_path):
        downloaded_bytes = os.path.getsize(output_path)
        
        # If we have the whole file
        if total_size > 0 and downloaded_bytes >= total_size:
            print(f"File {output_path} is already fully downloaded. Skipping.")
            return
            
        # If we have a partial file, resume from where we left off
        if downloaded_bytes > 0:
            print(f"Resuming {output_path} from {downloaded_bytes / (1024*1024):.2f} MB...")
            headers['Range'] = f"bytes={downloaded_bytes}-"
            mode = 'ab'
            initial_pos = downloaded_bytes

    print(f"Downloading {url} to {output_path}...")
    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    
    # If the server ignores the Range header, it will return 200 instead of 206
    if response.status_code == 200 and initial_pos > 0:
        print("Server doesn't support resuming. Restarting download from 0 bytes...")
        mode = 'wb'
        initial_pos = 0
        total_size = int(response.headers.get('content-length', 0))
    elif response.status_code == 206:
        # 206 Partial Content means resume worked
        if total_size == 0:
            # content-length for 206 is just the remaining bytes
            remaining = int(response.headers.get('content-length', 0))
            total_size = initial_pos + remaining

    with open(output_path, mode) as f, tqdm(
        desc=os.path.basename(output_path),
        initial=initial_pos,
        total=total_size if total_size > 0 else None,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
                
    print(f"Download complete: {output_path}")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # We don't extract if the target folder already has files, 
            # but FMA zips have a root folder inside them (e.g. fma_small/)
            folder_name = os.path.splitext(os.path.basename(zip_path))[0]
            if os.path.exists(os.path.join(extract_to, folder_name)):
                print(f"Folder {folder_name} already exists. Skipping extraction.")
                return
                
            for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                zip_ref.extract(member, extract_to)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print(f"ERROR: {zip_path} is corrupted. Please delete the file and run the script again to re-download it.")

def download_musiccaps_audio(csv_path, num_samples=50):
    """
    Downloads a subset of the MusicCaps audio from YouTube using yt-dlp.
    """
    out_dir = os.path.join(RAW_DIR, "musiccaps_audio")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Reading MusicCaps CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    subset = df.head(num_samples)
    print(f"Downloading audio for {num_samples} MusicCaps samples...")
    
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    
    for idx, row in tqdm(subset.iterrows(), total=len(subset)):
        ytid = row['ytid']
        start_time = row['start_s']
        end_time = row['end_s']
        
        out_file = os.path.join(out_dir, f"{ytid}.wav")
        if os.path.exists(out_file):
            continue
            
        cmd = [
            'python', '-m', 'yt_dlp',
            '-x', '--audio-format', 'wav',
            '--audio-quality', '0',
            '--ffmpeg-location', ffmpeg_path,
            '--postprocessor-args', f"-ar 16000 -ss {start_time} -to {end_time}",
            '-o', os.path.join(out_dir, f"%(id)s.%(ext)s"),
            f'https://www.youtube.com/watch?v={ytid}'
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to download video {ytid}")
            continue

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # 1. Download and Extract FMA Metadata
    metadata_zip = os.path.join(RAW_DIR, "fma_metadata.zip")
    download_file_resumable(FMA_METADATA_URL, metadata_zip)
    extract_zip(metadata_zip, RAW_DIR)
        
    # 2. Download and Extract FMA Small
    fma_small_zip = os.path.join(RAW_DIR, "fma_small.zip")
    download_file_resumable(FMA_SMALL_URL, fma_small_zip)
    extract_zip(fma_small_zip, RAW_DIR)
        
    # 3. Download MusicCaps CSV
    musiccaps_csv = os.path.join(RAW_DIR, "musiccaps-public.csv")
    download_file_resumable(MUSICCAPS_CSV_URL, musiccaps_csv)
    
    # 4. Download a subset of MusicCaps audio for Task 4
    try:
        download_musiccaps_audio(musiccaps_csv, num_samples=50)
    except Exception as e:
        print(f"Error downloading MusicCaps audio: {e}")
        
    print("\nAll downloads and extractions finished successfully!")

if __name__ == "__main__":
    main()

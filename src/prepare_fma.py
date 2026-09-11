import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def main():
    metadata_path = "data/raw/fma_metadata/tracks.csv"
    out_dir = "data/splits"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading metadata from {metadata_path}...")
    # FMA tracks.csv has a multi-level header
    tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1], low_memory=False)
    
    # Filter for small subset
    small = tracks[tracks[('set', 'subset')] == 'small']
    print(f"Found {len(small)} tracks in 'small' subset.")
    
    # The 8 top genres in fma_small
    top_genres = small[('track', 'genre_top')].unique().tolist()
    print(f"Classes ({len(top_genres)}): {top_genres}")
    
    mlb = MultiLabelBinarizer(classes=top_genres)
    mlb.fit([top_genres]) # Fit on all possible classes
    
    splits = {'training': [], 'validation': [], 'test': []}
    
    for track_id, row in small.iterrows():
        split_val = row[('set', 'split')] # 'training', 'validation', 'test'
        genre = row[('track', 'genre_top')]
        title = row[('track', 'title')]
        artist = row[('artist', 'name')]
        
        # Text proxy for Task 1 & 3
        # Handle nan values
        title_str = str(title) if pd.notna(title) else "Unknown Title"
        artist_str = str(artist) if pd.notna(artist) else "Unknown Artist"
        text_proxy = f"Title: {title_str}, Artist: {artist_str}"
        
        # One-hot encode the genre for BCE loss
        labels = mlb.transform([[genre]])[0].tolist()
        
        sample = {
            'track_id': f"{track_id:06d}",
            'text': text_proxy,
            'labels': labels,
            'genre': genre
        }
        
        if split_val in splits:
            splits[split_val].append(sample)
        else:
            splits['training'].append(sample) # Fallback just in case
            
    # Save the splits
    for split_name, data_list in splits.items():
        # Rename 'training' to 'train' and 'validation' to 'val' to match common conventions
        out_name = split_name
        if split_name == 'training': out_name = 'train'
        elif split_name == 'validation': out_name = 'val'
            
        out_path = os.path.join(out_dir, f"{out_name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(data_list)} samples to {out_path}")
        
    # Save class list
    with open(os.path.join(out_dir, "classes.json"), 'w') as f:
        json.dump(top_genres, f)

if __name__ == "__main__":
    main()

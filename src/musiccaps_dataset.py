import os
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from transformers import AutoTokenizer

class MusicCapsDataset(Dataset):
    def __init__(self, csv_path, tokenizer_name="distilbert-base-uncased", max_length=128, top_k_tags=50, split="train", subset_size=None):
        """
        MusicCaps Dataset for Task 1.
        Extracts `caption` as text input and `aspect_list` as multi-label proxy tags.
        """
        self.df = pd.read_csv(csv_path)
        
        # Parse aspect_list which is stored as string representation of list
        self.df['aspect_list'] = self.df['aspect_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        
        # Build tag vocabulary
        all_tags = []
        for tags in self.df['aspect_list']:
            # Lowercase and strip whitespace
            clean_tags = [str(t).lower().strip() for t in tags]
            all_tags.extend(clean_tags)
            
        tag_counts = Counter(all_tags)
        top_tags = [tag for tag, count in tag_counts.most_common(top_k_tags)]
        self.tag2idx = {tag: idx for idx, tag in enumerate(top_tags)}
        self.num_classes = len(self.tag2idx)
        
        # Assign fixed split manually since MusicCaps is a single file
        # 80% train, 20% val
        split_idx = int(len(self.df) * 0.8)
        if split == "train":
            self.df = self.df.iloc[:split_idx]
        else:
            self.df = self.df.iloc[split_idx:]
            
        if subset_size is not None:
            self.df = self.df.iloc[:subset_size]
            
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['caption'])
        tags = [str(t).lower().strip() for t in row['aspect_list']]
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Create multi-hot label vector
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        for tag in tags:
            if tag in self.tag2idx:
                label_vec[self.tag2idx[tag]] = 1.0
                
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_vec
        }

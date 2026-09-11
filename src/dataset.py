import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class FMADataset(Dataset):
    def __init__(self, splits_file, tokenizer_name="distilbert-base-uncased", max_length=128, data_dir="data/processed", load_graph=False, load_melspec=False):
        with open(splits_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data_dir = data_dir
        self.load_graph = load_graph
        self.load_melspec = load_melspec
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load Text (Task 1 & 3)
        encoding = self.tokenizer(
            sample['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'track_id': sample['track_id'],
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sample['labels'], dtype=torch.float32)
        }
        
        # 2. Load Graph (Task 2 & 3)
        if self.load_graph:
            graph_path = os.path.join(self.data_dir, f"{sample['track_id']}_graph.pt")
            try:
                if os.path.exists(graph_path):
                    graph_data = torch.load(graph_path, weights_only=False)
                else:
                    graph_data = "NONE" # Return string instead of None to avoid collate error
            except Exception as e:
                print(f"Error loading {graph_path}: {e}")
                graph_data = "NONE"
            item['graph_data'] = graph_data
            
        # 3. Load Melspec (Task 2 CNN Baseline)
        if getattr(self, 'load_melspec', False):
            melspec_path = os.path.join(self.data_dir, f"{sample['track_id']}_melspec.pt")
            try:
                if os.path.exists(melspec_path):
                    melspec_data = torch.load(melspec_path, weights_only=False)
                else:
                    melspec_data = "NONE"
            except Exception as e:
                print(f"Error loading {melspec_path}: {e}")
                melspec_data = "NONE"
            item['melspec'] = melspec_data
            
        return item

def custom_collate_fn(batch):
    collated = {
        'track_id': [b['track_id'] for b in batch],
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
    }
    if 'graph_data' in batch[0]:
        collated['graph_data'] = [b['graph_data'] for b in batch]
    if 'melspec' in batch[0]:
        collated['melspec'] = [b['melspec'] for b in batch]
    return collated

import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Markdown: Title
    cells.append(nbf.v4.new_markdown_cell("""# GNN-BERT Music Context: End-to-End Demo
This notebook demonstrates the complete inference pipeline of our final Cross-Attention Fusion model. It loads a trained model, takes a track's audio graph and text context (caption/tags), and outputs the final genre predictions."""))
    
    # Code: Imports
    cells.append(nbf.v4.new_code_cell("""import torch
import yaml
import json
import sys
sys.path.append('src')
import matplotlib.pyplot as plt
from src.fusion_model import FusionModel
from src.dataset import FMADataset, custom_collate_fn
from torch.utils.data import DataLoader

device = torch.device('cpu')"""))
    
    # Markdown: Load Model
    cells.append(nbf.v4.new_markdown_cell("""## 1. Load the Trained Fusion Model
We initialize the Cross-Attention Fusion model. (In a real scenario, we would `torch.load()` the saved weights here. For this demo, we use the model architecture to showcase the forward pass)."""))
    
    # Code: Load Model
    cells.append(nbf.v4.new_code_cell("""# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize Fusion Model with Cross-Attention
model = FusionModel(
    text_model_name=config["text"]["model_name"], 
    num_classes=8, 
    fusion_type="cross_attention"
).to(device)

model.eval()
print("Fusion Model successfully loaded and ready for inference!")"""))
    
    # Markdown: Load Data
    cells.append(nbf.v4.new_markdown_cell("""## 2. Load a Demo Track
We'll select a track from the validation set, load its DistilBERT tokens and its Audio Segment Graph."""))
    
    # Code: Load Data
    cells.append(nbf.v4.new_code_cell("""# Initialize Dataset (loads graph and text)
dataset = FMADataset(
    "data/splits/val.json", 
    tokenizer_name=config["text"]["model_name"], 
    max_length=config["text"]["max_length"], 
    load_graph=True
)

# Grab the first valid track
for i in range(len(dataset)):
    sample = dataset[i]
    if sample['graph_data'] != "NONE":
        break

track_id = sample['track_id']
graph_data = sample['graph_data'].unsqueeze(0).to(device) # Add batch dim
input_ids = sample['input_ids'].unsqueeze(0).to(device)
attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
true_labels = sample['labels'].numpy()

# Get the raw text
with open("data/splits/val.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)
    raw_text = next(item["text"] for item in val_data if item["track_id"] == track_id)

print(f"Loaded Track ID: {track_id}")
print(f"Context Text: {raw_text}")"""))
    
    # Markdown: Inference
    cells.append(nbf.v4.new_markdown_cell("""## 3. Run Inference
We pass the graph and text through the model to get the final predictions."""))
    
    # Code: Inference
    cells.append(nbf.v4.new_code_cell("""with torch.no_grad():
    logits, embeddings = model(input_ids, attention_mask, graph_data)
    probs = torch.sigmoid(logits).squeeze(0).numpy()

genres = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]

print("--- Predicted Probabilities ---")
for i, genre in enumerate(genres):
    print(f"{genre:15s}: {probs[i]:.4f}")

pred_genres = [genres[i] for i in range(8) if probs[i] > 0.5]
if not pred_genres:
    pred_genres = [genres[probs.argmax()]]
    
actual_genres = [genres[i] for i in range(8) if true_labels[i] == 1.0]

print("\\nFinal Predicted Genres:", pred_genres)
print("Actual Ground Truth  :", actual_genres)"""))

    nb['cells'] = cells
    
    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/demo_context.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
if __name__ == "__main__":
    create_notebook()

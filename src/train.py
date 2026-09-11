import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import FMADataset, custom_collate_fn
from musiccaps_dataset import MusicCapsDataset
from bert_encoder import BERTEncoder
from gnn_model import AudioGNN
from cnn_baseline import MelCNN
from fusion_model import FusionModel
from evaluate import evaluate_model, evaluate_random_baseline, plot_curves

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def train_task1(config, device):
    print("--- Training Task 1: BERT Text Classifier on MusicCaps ---")
    
    train_dataset = MusicCapsDataset("data/raw/musiccaps-public.csv", tokenizer_name=config["text"]["model_name"], max_length=config["text"]["max_length"], split="train", subset_size=320)
    val_dataset = MusicCapsDataset("data/raw/musiccaps-public.csv", tokenizer_name=config["text"]["model_name"], max_length=config["text"]["max_length"], split="val", subset_size=80)
    
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    model = BERTEncoder(model_name=config["text"]["model_name"], num_classes=50).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    
    epochs = config["training"]["epochs"]
    history = {'loss': [], 'macro_f1': [], 'micro_f1': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits, _, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        plot_pr = (epoch == epochs - 1)
        metrics = evaluate_model(model, val_loader, device, task=1, plot_pr=plot_pr)
        print(f"Val Metrics: {metrics}")
        
        history['loss'].append(avg_loss)
        history['macro_f1'].append(metrics['Macro-F1'])
        history['micro_f1'].append(metrics['Micro-F1'])
        
    plot_curves(history, task=1)

def train_task2(config, device):
    print("--- Training Task 2: Baseline Comparison (CNN vs GNN) ---")
    
    train_dataset = FMADataset("data/splits/train.json", load_graph=True, load_melspec=True)
    val_dataset = FMADataset("data/splits/val.json", load_graph=True, load_melspec=True)
    
    train_dataset.samples = train_dataset.samples[:320]
    val_dataset.samples = val_dataset.samples[:80]
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    # Random Baseline
    print(">>> Random Baseline:")
    # We just run one batch to get the shape of labels
    for batch in val_loader:
        labels = batch['labels'].numpy()
        break
    import numpy as np
    y_true_mock = np.zeros((len(val_dataset), 8))
    for i, b in enumerate(val_loader):
        y_true_mock[i] = b['labels'].numpy()
    rand_metrics = evaluate_random_baseline(y_true_mock)
    print(f"Random Baseline Metrics: {rand_metrics}")

    # CNN
    print("\n>>> Training CNN Baseline")
    cnn_model = MelCNN(out_channels=8).to(device)
    cnn_opt = optim.Adam(cnn_model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    cnn_history = {'loss': [], 'macro_f1': [], 'micro_f1': []}
    epochs = config["training"]["epochs"]
    for epoch in range(epochs):
        cnn_model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            melspec = batch['melspec']
            if melspec[0] == "NONE": continue
            melspec = melspec[0].to(device)
            labels = batch['labels'].to(device)
            
            cnn_opt.zero_grad()
            logits, _ = cnn_model(melspec)
            loss = criterion(logits, labels)
            loss.backward()
            cnn_opt.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss/len(train_loader)
        plot_pr = (epoch == epochs - 1)
        metrics = evaluate_model(cnn_model, val_loader, device, task=2, plot_pr=plot_pr)
        print(f"CNN Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Macro-F1: {metrics['Macro-F1']:.4f}")
        
        cnn_history['loss'].append(avg_loss)
        cnn_history['macro_f1'].append(metrics['Macro-F1'])
        cnn_history['micro_f1'].append(metrics['Micro-F1'])
        
    plot_curves(cnn_history, task=2) # Will save as task2_curves.png
    
    # GNN
    print("\n>>> Training GNN Baseline")
    gnn_model = AudioGNN(out_channels=8).to(device)
    gnn_opt = optim.Adam(gnn_model.parameters(), lr=0.001)
    
    gnn_history = {'loss': [], 'macro_f1': [], 'micro_f1': []}
    for epoch in range(epochs):
        gnn_model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            graph_data = batch['graph_data']
            if graph_data[0] == "NONE": continue
            graph_data = graph_data[0].to(device)
            labels = batch['labels'].to(device)
            
            gnn_opt.zero_grad()
            logits, _ = gnn_model(graph_data)
            loss = criterion(logits.squeeze(0), labels.squeeze(0))
            loss.backward()
            gnn_opt.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss/len(train_loader)
        plot_pr = (epoch == epochs - 1)
        metrics = evaluate_model(gnn_model, val_loader, device, task=3, plot_pr=plot_pr) # task=3 in evaluate is GNN
        print(f"GNN Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Macro-F1: {metrics['Macro-F1']:.4f}")
        
        gnn_history['loss'].append(avg_loss)
        gnn_history['macro_f1'].append(metrics['Macro-F1'])
        gnn_history['micro_f1'].append(metrics['Micro-F1'])

    plot_curves(gnn_history, task=3) # GNN saved as task3_curves.png
    
    # We can plot comparison manually if needed, but saving them separately is fine.

def train_task3(config, device):
    print("--- Training Task 3: GNN-BERT Fusion Model ---")
    
    train_dataset = FMADataset("data/splits/train.json", load_graph=True)
    val_dataset = FMADataset("data/splits/val.json", load_graph=True)
    
    train_dataset.samples = train_dataset.samples[:320]
    val_dataset.samples = val_dataset.samples[:80]
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    
    model = FusionModel(text_model_name=config["text"]["model_name"], num_classes=8, fusion_type="cross_attention").to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    
    epochs = config["training"]["epochs"]
    history = {'loss': [], 'macro_f1': [], 'micro_f1': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            graph_data = batch['graph_data']
            if graph_data[0] == "NONE": continue
            graph_data = graph_data[0].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask, graph_data)
            loss = criterion(logits.squeeze(0), labels.squeeze(0))
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        plot_pr = (epoch == epochs - 1)
        metrics = evaluate_model(model, val_loader, device, task=4, plot_pr=plot_pr)
        print(f"Val Metrics: {metrics}")
        
        history['loss'].append(avg_loss)
        history['macro_f1'].append(metrics['Macro-F1'])
        history['micro_f1'].append(metrics['Micro-F1'])
        
    plot_curves(history, task=4)

def main():
    parser = argparse.ArgumentParser(description="Train GNN-BERT Music Context Models")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], required=True, help="Task number to train")
    args = parser.parse_args()
    
    config = load_config()
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.task == 1:
        train_task1(config, device)
    elif args.task == 2:
        train_task2(config, device)
    elif args.task == 3:
        train_task3(config, device)

if __name__ == "__main__":
    main()

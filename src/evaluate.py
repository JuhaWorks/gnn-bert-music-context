import os
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_curve, auc
import numpy as np
import matplotlib.pyplot as plt

def calculate_macro_f1(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs > threshold).astype(int)
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def calculate_micro_f1(y_true, y_pred_probs, threshold=0.5):
    y_pred = (y_pred_probs > threshold).astype(int)
    return f1_score(y_true, y_pred, average='micro', zero_division=0)

def calculate_auc_pr(y_true, y_pred_probs, task=None, plot=False):
    num_classes = y_true.shape[1]
    auc_pr_scores = []
    genre_names = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]
    
    if plot and task is not None:
        os.makedirs("plots", exist_ok=True)
    
    for i in range(num_classes):
        if sum(y_true[:, i]) == 0:
            continue
            
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
        class_auc_pr = auc(recall, precision)
        auc_pr_scores.append(class_auc_pr)
        
        if plot and task is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, 'b-', label=f'AUC = {class_auc_pr:.4f}')
            plt.title(f'Task {task}: PR Curve - {genre_names[i] if i < len(genre_names) else f"Class {i}"}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"plots/task{task}_pr_curve_class_{i}.png")
            plt.close()
        
    return np.mean(auc_pr_scores) if auc_pr_scores else 0.0

def evaluate_random_baseline(y_true):
    # Generates random probabilities and calculates metrics
    # Shape of y_true: (num_samples, num_classes)
    y_pred_probs = np.random.rand(*y_true.shape)
    macro_f1 = calculate_macro_f1(y_true, y_pred_probs)
    micro_f1 = calculate_micro_f1(y_true, y_pred_probs)
    auc_pr = calculate_auc_pr(y_true, y_pred_probs)
    
    return {
        "Macro-F1": macro_f1,
        "Micro-F1": micro_f1,
        "AUC-PR": auc_pr
    }

def plot_tsne(embeddings, labels, task=3):
    from sklearn.manifold import TSNE
    os.makedirs("plots", exist_ok=True)
    
    # We map the multi-hot labels to a primary class just for coloring
    primary_classes = np.argmax(labels, axis=1)
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=primary_classes, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Primary Genre")
    plt.title(f't-SNE Visualization of Embeddings (Task {task})')
    plt.savefig(f"plots/task{task}_tsne.png")
    plt.close()

def evaluate_model(model, dataloader, device, task=1, plot_pr=False):
    model.eval()
    all_y_true = []
    all_y_pred_probs = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            y_true = batch['labels'].to(device)
            
            if task == 1:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits, _, _ = model(input_ids, attention_mask)
                emb = logits # Placeholder
            elif task == 2: # CNN
                melspec = batch['melspec']
                if melspec[0] == "NONE": continue
                melspec = melspec[0].to(device)
                logits, x = model(melspec)
                emb = x
            elif task == 3: # GNN
                graph_data = batch['graph_data']
                if graph_data[0] == "NONE": continue
                graph_data = graph_data[0].to(device)
                logits, g = model(graph_data)
                emb = g
            elif task == 4: # Fusion
                graph_data = batch['graph_data']
                if graph_data[0] == "NONE": continue
                graph_data = graph_data[0].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits, z = model(input_ids, attention_mask, graph_data)
                emb = z
                
            probs = torch.sigmoid(logits)
            
            all_y_true.append(y_true.cpu().numpy())
            all_y_pred_probs.append(probs.cpu().numpy())
            if task > 1:
                all_embeddings.append(emb.cpu().numpy())
            
    all_y_true = np.vstack(all_y_true)
    all_y_pred_probs = np.vstack(all_y_pred_probs)
    if task > 1:
        all_embeddings = np.vstack(all_embeddings)
        if plot_pr:
            plot_tsne(all_embeddings, all_y_true, task=task)
    
    macro_f1 = calculate_macro_f1(all_y_true, all_y_pred_probs)
    micro_f1 = calculate_micro_f1(all_y_true, all_y_pred_probs)
    auc_pr = calculate_auc_pr(all_y_true, all_y_pred_probs, task=task, plot=plot_pr)
    
    return {
        "Macro-F1": macro_f1,
        "Micro-F1": micro_f1,
        "AUC-PR": auc_pr
    }

def plot_curves(history, task):
    os.makedirs("plots", exist_ok=True)
    epochs = range(1, len(history['loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'bo-', label='Training Loss')
    plt.title(f'Task {task}: Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # F1 plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['macro_f1'], 'ro-', label='Val Macro-F1')
    plt.plot(epochs, history['micro_f1'], 'go-', label='Val Micro-F1')
    plt.title(f'Task {task}: F1 Scores over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"plots/task{task}_learning_curves.png")
    plt.close()

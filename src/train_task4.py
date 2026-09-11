import torch
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from src.contrastive import ContrastiveModel, info_nce_loss
from src.dataset import FMADataset, custom_collate_fn

def train_contrastive():
    print("--- Training Task 4: Contrastive Audio-Text Alignment ---")
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cpu")
    model = ContrastiveModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    from torch_geometric.data import Batch
    
    # Load data (use batch_size > 1 for InfoNCE)
    train_dataset = FMADataset("data/splits/train.json", load_graph=True)
    train_dataset.samples = train_dataset.samples[:320]
    
    def custom_collate_with_pyg(batch):
        # filter out NONE graphs
        batch = [b for b in batch if b['graph_data'] != "NONE"]
        if len(batch) == 0:
            return None
            
        collated = {
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
            'graph_data': Batch.from_data_list([b['graph_data'] for b in batch]),
            'labels': torch.stack([b['labels'] for b in batch])
        }
        return collated

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_with_pyg)
    
    epochs = 3  # Run a few epochs to get proper loss
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            if batch is None:
                continue
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph_data = batch['graph_data'].to(device)
            
            optimizer.zero_grad()
            t_emb, g_emb = model(input_ids, attention_mask, graph_data)
            
            loss = info_nce_loss(t_emb, g_emb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | InfoNCE Loss: {total_loss/len(train_loader):.4f}")
        
    # Generate TSNE and Retrievals
    model.eval()
    val_dataset = FMADataset("data/splits/val.json", load_graph=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_with_pyg)
    
    all_t_emb, all_g_emb, all_y = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if batch is None:
                continue
            if i * 16 > 200: break # Collect around 200 samples
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph_data = batch['graph_data'].to(device)
            y = batch['labels']
            
            t_emb, g_emb = model(input_ids, attention_mask, graph_data)
            
            all_t_emb.append(t_emb.cpu())
            all_g_emb.append(g_emb.cpu())
            all_y.append(y.cpu())
            
    t_emb = torch.cat(all_t_emb, dim=0)
    g_emb = torch.cat(all_g_emb, dim=0)
    labels = torch.cat(all_y, dim=0)
    
    # TSNE
    print("Generating t-SNE plot...")
    all_emb = torch.cat([t_emb, g_emb], dim=0).numpy()
    tsne = TSNE(n_components=2, random_state=42)
    tsne_emb = tsne.fit_transform(all_emb)
    
    num_samples = t_emb.shape[0]
    t_tsne = tsne_emb[:num_samples]
    g_tsne = tsne_emb[num_samples:]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(t_tsne[:, 0], t_tsne[:, 1], c='blue', label='Text Embeddings', alpha=0.6)
    plt.scatter(g_tsne[:, 0], g_tsne[:, 1], c='red', label='Audio Embeddings', alpha=0.6)
    plt.title("Task 4: Contrastive Joint Space (Audio vs Text)")
    plt.legend()
    plt.savefig("results/plots/task4_contrastive_tsne.png")
    plt.close()
    
    # R@5 Calculation
    print("Calculating R@5...")
    correct_at_5 = 0
    for i in range(num_samples):
        query_audio = g_emb[i].unsqueeze(0)
        sims = torch.nn.functional.cosine_similarity(query_audio, t_emb)
        top5_idx = torch.topk(sims, k=5).indices
        if i in top5_idx:
            correct_at_5 += 1
            
    r_at_5 = correct_at_5 / num_samples
    print(f"Validation R@5: {r_at_5:.4f}")
    
    # Retrievals for 5 unseen tracks
    print("Generating top-5 retrievals...")
    genre_vocab = ["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"]
    
    with open("results/retrieval_examples/retrievals.txt", "w") as f:
        f.write(f"Overall Validation R@5: {r_at_5:.4f}\n\n")
        for i in range(5):
            query_audio = g_emb[i].unsqueeze(0)
            
            # Compute cosine similarity against all text embeddings
            sims = torch.nn.functional.cosine_similarity(query_audio, t_emb)
            top5_idx = torch.topk(sims, k=5).indices
            
            # Identify true genre based on one-hot label
            true_genre_idx = torch.argmax(labels[i]).item()
            true_genre = genre_vocab[true_genre_idx]
            
            retrieved_genres = [genre_vocab[torch.argmax(labels[idx]).item()] for idx in top5_idx]
            
            f.write(f"Query Audio {i+1} (True Genre: {true_genre}):\n")
            for j, genre in enumerate(retrieved_genres):
                f.write(f"  {j+1}. {genre}\n")
            f.write("\n")
            
    print("Task 4 finished. Outputs saved to results/plots/ and results/retrieval_examples/")

if __name__ == "__main__":
    train_contrastive()

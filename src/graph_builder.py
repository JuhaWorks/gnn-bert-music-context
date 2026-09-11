import os
import glob
import yaml
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from tqdm import tqdm

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def build_graph(features_tensor, threshold):
    """
    Builds a PyTorch Geometric graph from sequence of segment features.
    Nodes: 5-second audio segments.
    Edges: Temporal adjacency + Cosine similarity > threshold.
    """
    num_nodes = features_tensor.shape[0]
    if num_nodes < 2:
        return None
        
    edge_index = []
    
    # 1. Temporal Adjacency Edges (connect i to i+1)
    for i in range(num_nodes - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i]) # Bidirectional
        
    # 2. Cosine Similarity Edges
    # Compute pairwise cosine similarity
    normalized_features = F.normalize(features_tensor, p=2, dim=1)
    sim_matrix = torch.matmul(normalized_features, normalized_features.T)
    
    for i in range(num_nodes):
        for j in range(i + 2, num_nodes): # Skip adjacent nodes to avoid duplicates
            if sim_matrix[i, j] > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
                
    if not edge_index:
        return None
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Create PyTorch Geometric Data object
    data = Data(x=features_tensor, edge_index=edge_index)
    return data

def main():
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    threshold = config["graph"]["similarity_threshold"]
    
    feature_files = glob.glob(os.path.join(processed_dir, "*_features.pt"))
    print(f"Found {len(feature_files)} feature files. Building graphs...")
    
    for filepath in tqdm(feature_files):
        track_id = os.path.basename(filepath).split("_")[0]
        out_path = os.path.join(processed_dir, f"{track_id}_graph.pt")
        
        if os.path.exists(out_path):
            continue
            
        features = torch.load(filepath)
        graph_data = build_graph(features, threshold)
        
        if graph_data is not None:
            torch.save(graph_data, out_path)

if __name__ == "__main__":
    main()

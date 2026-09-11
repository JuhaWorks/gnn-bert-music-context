import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

class AudioGNN(nn.Module):
    def __init__(self, in_channels=12, hidden_channels=64, out_channels=8):
        super(AudioGNN, self).__init__()
        # GraphSAGE layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Classification head
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Global mean pooling to get a single vector per graph
        g = global_mean_pool(x, batch)
        
        logits = self.classifier(g)
        # Remove sigmoid here, BCEWithLogitsLoss expects raw logits
        
        return logits, g

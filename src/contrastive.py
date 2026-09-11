import torch
import torch.nn as nn
import torch.nn.functional as F
from .bert_encoder import BERTEncoder
from .gnn_model import AudioGNN

class ContrastiveModel(nn.Module):
    def __init__(self, text_model_name="distilbert-base-uncased", temperature=0.07):
        super(ContrastiveModel, self).__init__()
        self.bert_encoder = BERTEncoder(model_name=text_model_name, num_classes=8)
        self.gnn_encoder = AudioGNN(out_channels=8)
        
        self.temperature = temperature
        
        # Projection heads to bring both to the same embedding space (e.g., 128 dim)
        self.text_proj = nn.Linear(768, 128)
        self.audio_proj = nn.Linear(64, 128)

    def forward(self, input_ids, attention_mask, graph_data):
        # Text embeddings
        _, t, _ = self.bert_encoder(input_ids, attention_mask)
        t_emb = F.normalize(self.text_proj(t), p=2, dim=1)
        
        # Audio embeddings
        _, g = self.gnn_encoder(graph_data)
        g_emb = F.normalize(self.audio_proj(g), p=2, dim=1)
        
        return t_emb, g_emb

def info_nce_loss(t_emb, g_emb, temperature=0.07):
    """
    Computes InfoNCE loss for a batch of paired text and audio embeddings.
    """
    batch_size = t_emb.size(0)
    
    # Calculate similarity matrix (Audio x Text)
    sim_matrix = torch.matmul(g_emb, t_emb.T) / temperature
    
    # Labels are just the diagonal (i.e. i-th audio matches i-th text)
    labels = torch.arange(batch_size).to(t_emb.device)
    
    # Loss in both directions
    loss_a2t = F.cross_entropy(sim_matrix, labels)
    loss_t2a = F.cross_entropy(sim_matrix.T, labels)
    
    return (loss_a2t + loss_t2a) / 2

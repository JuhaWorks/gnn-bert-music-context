import torch
import torch.nn as nn
import math
from bert_encoder import BERTEncoder
from gnn_model import AudioGNN

class FusionModel(nn.Module):
    def __init__(self, text_model_name="distilbert-base-uncased", num_classes=8, fusion_type='cross_attention'):
        super(FusionModel, self).__init__()
        self.fusion_type = fusion_type
        self.bert_encoder = BERTEncoder(model_name=text_model_name, num_classes=num_classes)
        self.gnn_encoder = AudioGNN(out_channels=num_classes)
        
        # Assume distilbert output is 768, and GNN output is 64
        self.bert_dim = 768
        self.gnn_dim = 64
        self.d_k = 64 # dimension for query/key
        
        if self.fusion_type == 'cross_attention':
            self.W_Q = nn.Linear(self.gnn_dim, self.d_k)
            self.W_K = nn.Linear(self.bert_dim, self.d_k)
            self.W_V = nn.Linear(self.bert_dim, self.bert_dim)
            
            self.fusion_classifier = nn.Sequential(
                nn.Linear(self.gnn_dim + self.bert_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        else:
            # Simple Concatenation Fusion
            self.fusion_classifier = nn.Sequential(
                nn.Linear(self.bert_dim + self.gnn_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
            
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, graph_data):
        # 1. Text Representation
        _, t, H_text = self.bert_encoder(input_ids, attention_mask)
        # H_text shape: (Batch, SeqLen, 768)
        
        # 2. Graph Representation
        _, g = self.gnn_encoder(graph_data)
        # g shape: (Batch, 64)
        
        # 3. Fusion
        if self.fusion_type == 'cross_attention':
            # Q = g * W_Q -> (Batch, d_k)
            Q = self.W_Q(g).unsqueeze(1) # (Batch, 1, d_k)
            
            # K = H_text * W_K -> (Batch, SeqLen, d_k)
            K = self.W_K(H_text)
            
            # V = H_text * W_V -> (Batch, SeqLen, 768)
            V = self.W_V(H_text)
            
            # Attention scores: A = softmax(QK^T / sqrt(d_k))
            # Q * K^T -> (Batch, 1, SeqLen)
            scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)
            
            # Mask out padding tokens
            mask = attention_mask.unsqueeze(1) # (Batch, 1, SeqLen)
            scores = scores.masked_fill(mask == 0, -1e9)
            
            A = torch.softmax(scores, dim=-1)
            
            # Context vector: A * V -> (Batch, 1, 768)
            context = torch.bmm(A, V).squeeze(1) # (Batch, 768)
            
            # Final representation: z = CONCAT(g, context)
            z = torch.cat([g, context], dim=1) # (Batch, 64 + 768)
        else:
            z = torch.cat([g, t], dim=1)
        
        logits = self.fusion_classifier(z)
        probs = self.sigmoid(logits)
        
        return probs, z

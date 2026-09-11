import torch
import torch.nn as nn

class MelCNN(nn.Module):
    def __init__(self, out_channels=8):
        super(MelCNN, self).__init__()
        # Input shape: (Batch, 1, 128, T)
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 x T/2
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32 x T/4
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 x T/8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8 x T/16
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8, 256), # Time dimension will be pooled
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, out_channels)
        )

    def forward(self, x):
        # x is (Batch, 128, T) or (128, T)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0) # -> (1, 1, 128, T)
        elif x.dim() == 3:
            x = x.unsqueeze(1) # Add channel dimension -> (Batch, 1, 128, T)
            
        x = self.features(x)
        
        # Adaptive pooling over time dimension to size 1
        # Shape before: (Batch, 128, 8, T')
        x = torch.mean(x, dim=-1) # Global average pooling over time -> (Batch, 128, 8)
        x = x.view(x.size(0), -1) # Flatten -> (Batch, 128*8)
        
        logits = self.classifier(x)
        return logits, x

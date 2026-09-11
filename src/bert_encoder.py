import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BERTEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=8):
        super(BERTEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT for low PC memory optimization if needed
        # for param in self.bert.parameters():
        #     param.requires_grad = False
            
        config = AutoConfig.from_pretrained(model_name)
        hidden_size = config.hidden_size # usually 768
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT doesn't have pooler_output, so we take the first token [CLS]
        last_hidden_state = outputs.last_hidden_state
        cls_token_state = last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_token_state)
        # Remove sigmoid here, use BCEWithLogitsLoss during training
        
        return logits, cls_token_state, last_hidden_state

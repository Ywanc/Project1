from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn

class TextClassifier(nn.Module): # of parameters: 109712516 (LegalBERT + FCL)
    def __init__(self, num_cats):
        super().__init__()
        
        # legalbert
        self.legalbert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.hidden_size = self.legalbert.config.hidden_size # embedding dimension
        
        # freeze first 6 layers of legalbert 
        for param in self.legalbert.encoder.layer[:6].parameters():
            param.requires_grad = False
        
        # Fully connected layers (classifier)
        self.layer1 = nn.Linear(self.hidden_size, 128)  # Increased from 256 to 512
        self.layer2 = nn.Linear(128, 64)                # Increased from 128 to 256            # New layer
        self.outlayer = nn.Linear(64, num_cats)

        # activation function & dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_ids, attention_mask):
        # [batch_size, seq_len, hidden_size]
        outputs = self.legalbert(input_ids=input_ids, attention_mask=attention_mask) # token ids, masking for padding tokens
        
        # [batch_size, hidden_size], so one vector of hidden_size per sample
        cls_output = outputs.last_hidden_state[:, 0, :] # get cls token (first token)
        
        # pass the cls token to the NN
        x = self.dropout(self.relu(self.layer1(cls_output)))
        x = self.dropout(self.relu(self.layer2(x)))
        logits = self.outlayer(x)
        
        return logits
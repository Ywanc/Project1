from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn

class LegalBERTClassifier(nn.module):
    def __init__(self):
        super().__init__()
        
        #legalbert
        self.legalbert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased") 
        self.hidden_size = self.legalbert.config.hidden_size
        
        # Classifier Head
        
    
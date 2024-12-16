import shutil
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")




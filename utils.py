import shutil
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
text = "testing"
tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus", use_fast=False)  
sum_model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus").to(device)

input_tokenized = tokenizer.encode(text, return_tensors='pt',max_length=1024,truncation=True)
summary_ids = sum_model.generate(input_tokenized,
                                  num_beams=5,
                                  no_repeat_ngram_size=3,
                                  length_penalty=4.0,
                                  min_length=150,
                                  max_length=500,
                                  early_stopping=True)
summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
print(f"Initial Summary: {summary}")
print(f"Summary word count: {len(summary.split())}")
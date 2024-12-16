import torch
from classifier.classify import classify, scrape_pars
from classifier.model import TextClassifier
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizer, BartForConditionalGeneration
import os 
import shutil
import gdown

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Classify factual paragraphs
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
clf_model = TextClassifier(num_cats=3).to(device)

# download weights 
url = 'https://drive.google.com/uc?id=12avyewxhHAW9NoveVD1KBVvGiM5qp40r'

output_path = 'clf_weights.pth' 
try:
    gdown.download(url, output_path, quiet=False)
    print("Download complete")
except Exception as e:
    print(f"Failed to download: {e}")
    
clf_model.load_state_dict(torch.load("clf_weights.pth", map_location=device))

url = input("Elit Link: ")
paragraphs = scrape_pars(url)
facts = []
for par in paragraphs:
    logits, conf, pred = classify(par, clf_model, tokenizer, device)
    if pred == "Facts":
        facts.append(par)
facts_text = " ".join(facts)
print(facts_text)
print(f"Word Count: {len(facts_text.split())}")


# 2) Summarise the factual paragraphs
# LegalPegasus
sum_tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")  
sum_model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus").to(device)

input_tokenized = sum_tokenizer.encode(facts_text, return_tensors='pt',max_length=1024,truncation=True).input_ids.to(device)
summary_ids = sum_model.generate(input_tokenized,
                                  num_beams=5,
                                  no_repeat_ngram_size=3,
                                  length_penalty=4.0,
                                  min_length=150,
                                  max_length=500,
                                  early_stopping=True)
summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(f"Initial Summary: {summary}")
print(f"Summary word count: {len(summary.split())}")

"""
sum_tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-led-base-16384")  
sum_model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-led-base-16384")

input_tokenized = sum_tokenizer.encode(facts_text, return_tensors='pt',padding="max_length",pad_to_max_length=True, max_length=6144,truncation=True)
summary_ids = sum_model.generate(input_tokenized,
                                  num_beams=4,
                                  no_repeat_ngram_size=3,
                                  length_penalty=2,
                                  min_length=350,
                                  max_length=500)

summary = [sum_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
print(summary)"""

# 3) Refine the summary
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
prompt = f"Refine the following summary to focus only on the facts of the case and improve its readability: {summary}"

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
outputs = model.generate(input_ids, max_length = 350, num_beams=5, early_stopping=True)

refined_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Refined summary: {refined_summary}")
print(f"Word Count: {len(refined_summary.split())}")

"""
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

prompt = f"Refine the following text to make it shorter, more cohesive and reader-friendly: {summary}"
inputs = bart_tokenizer(prompt, return_tensors="pt")

outputs = bart_model.generate(inputs["input_ids"], max_length=350, num_beams=5, length_penalty=4, early_stopping=True)
refined_summary = bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Refined Summary: {refined_summary}")
print(f"Word Count: {len(refined_summary.split())}")"""
import torch
from classifier.classify import scrape_pars
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizer, BartForConditionalGeneration
from setfit import SetFitModel
import gdown, os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 0) Download model weights
weights_path = 'trained_setfit_model'
if not os.path.exists(weights_path):
    weights_url = "https://drive.google.com/drive/folders/1LaHFXej2nYgdGmjwsjnIuD4_mv3_JToS"
    gdown.download_folder(weights_url)
else:
    print(f"File {weights_path} already exists. Skipping download.")

# 1) Classify factual paragraphs
clf_model = SetFitModel.from_pretrained("trained_setfit_model")
clf_model.to(device)
url = input("Elit Link: ")
pars = scrape_pars(url)
preds = clf_model.predict(pars)

full_facts, i = [], 1
for par, pred in zip(pars, preds):
    text = " ".join(par.split()[:6])
    print(f"Paragraph {i} : {text}...\nPredicted label: {pred}\n")
    i+=1
    if pred == 1:
        full_facts.append(par)
facts_text = " ".join(full_facts)
print(f"Extracted Facts: {facts_text}")
print(f"Word Count: {len(facts_text.split())}")


# 2) Summarise the factual paragraphs
# LegalPegasus
sum_tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus")  
sum_model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus").to(device)

input_tokenized = sum_tokenizer.encode_plus(facts_text, return_tensors='pt', max_length=1024, truncation=True)
input_ids = input_tokenized['input_ids'].to(device)

summary_ids = sum_model.generate(input_ids,
                                  num_beams=5,
                                  no_repeat_ngram_size=3,
                                  length_penalty=4.0,
                                  min_length=150,
                                  max_length=500,
                                  early_stopping=True)
summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(f"Initial Summary: {summary}")
print(f"Summary word count: {len(summary.split())}")

# 3) Refine the summary
further_summary = ""
while further_summary != "Y" and further_summary != "N":
    further_summary = input("Do you want to summarise further? (Y/N)")
if further_summary == "N":
    exit()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
prompt = f"Summarize the following text to make it more concise: {summary}"

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
outputs = model.generate(input_ids, min_length=128, max_length = 512, num_beams=5, early_stopping=True)

refined_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Refined summary: {refined_summary}")
print(f"Word Count: {len(refined_summary.split())}")

"""
summary_tokenized = sum_tokenizer.encode_plus(summary, return_tensors='pt', max_length=1024, truncation=True)
input_ids = summary_tokenized['input_ids'].to(device)
summary_ids = sum_model.generate(input_ids,
                                  num_beams=5,
                                  no_repeat_ngram_size=3,
                                  length_penalty=10.0,
                                  max_length=256,
                                  early_stopping=True)
refined_summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(f"Initial Summary: {refined_summary}")
print(f"Summary word count: {len(refined_summary.split())}")"""

""" # BART
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)

prompt = f"Summarize this text to make it clearer and more concise: {summary}"
inputs_ids = bart_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)

outputs = bart_model.generate(inputs_ids, max_length=350, num_beams=5, length_penalty=10, early_stopping=True)
refined_summary = bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Refined Summary: {refined_summary}")
print(f"Word Count: {len(refined_summary.split())}")"""
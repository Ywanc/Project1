import torch
from setfit import SetFitModel
import re
import requests
from bs4 import BeautifulSoup

# takes in text and classifies it 
def classify(input_text, model, tokenizer, device):
    num_to_label = {0: 'Facts',1: 'Decision', 2: 'Others'}

    tokenized_input = tokenizer(input_text, padding=True, truncation=True,return_tensors='pt')
    input_ids = tokenized_input['input_ids'].to(device)
    attention_mask = tokenized_input['attention_mask'].to(device)
    # make prediction with model
    model.eval()
    with torch.inference_mode():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        confidence = torch.max(probs, dim=1).values
        pred = torch.argmax(probs, dim=1)
    return logits, round(confidence.item(), 3), num_to_label[int(pred.item())]

# takes elit link, returns list of all the paragraphs
def scrape_pars(url):
    def remove_footnotes(content):
        for sup in content.find_all('sup'): 
            sup.decompose()  
        for modal in content.find_all('div', class_=lambda x: x and "modal fade" in x):
            modal.decompose()
        return " ".join(content.stripped_strings)

    def remove_num(text):
        return re.sub(r'^\d+\s+', '', text, flags=re.MULTILINE)
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all(lambda tag: tag.name in ['div', 'p'] and ('Judg-1' in (tag.get('class', [])) if tag.has_attr('class') else False))

    par_list = []
    for par in paragraphs:
        # curr text
        cur_tag_text = remove_footnotes(par).strip()
        cur_tag_text = re.sub(r'\s+', ' ', remove_num(cur_tag_text)).strip()
        par_list.append(cur_tag_text)
        
        # check for any children paragraph
        for sibling in par.find_next_siblings():
            
            # if reach any headers stop
            if sibling.has_attr('class') and any('Judg-Heading' in x for x in sibling['class']): 
                break
            cur_tag_text = remove_footnotes(sibling).strip()
            # if reach new paragraph stop
            if bool(re.match(r'^\d', cur_tag_text)):
                break
            # if tag has no number in front, & is judg-x, then add it
            if sibling.has_attr('class') and any(re.search(r'Judg-\d', cls) for cls in sibling['class']):
                cur_tag_text = re.sub(r'\s+', ' ', remove_num(remove_footnotes(sibling))).strip()
                par_list[-1] = par_list[-1] + " " + cur_tag_text
    return par_list

# Get Url
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SetFitModel.from_pretrained("train_results\\trained_model")
model.to(device)

url = input("Elit Link: ")
print(" ")
pars = scrape_pars(url)

preds = model.predict(pars)

i=1
full_facts = []
for par, pred in zip(pars, preds):
    text = " ".join(par.split()[:6])
    print(f"Paragraph {i} : {text}...\nPredicted label: {pred}\n")
    i+=1
    if pred == 1:
        full_facts.append(par)
    
print(" ".join(full_facts))
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = TextClassifier(num_cats=3).to(device)
model.load_state_dict(torch.load("weights\\best_model_weights.pth", map_location=device, weights_only=True))

url = input("Elit Link: ")
pars = scrape_pars(url)

facts = []
i = 1
for par in pars:
    logits, conf, pred = classify(par, model, tokenizer, device)
    print(f"{i}: {pred}")
    if pred == "Facts":
        facts.append(par)
    i+=1
print(facts)

"""

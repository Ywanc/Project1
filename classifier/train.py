import pandas as pd
import os
import torch 
from sklearn.model_selection import train_test_split
from datasets import Dataset
from setfit import SetFitModel
from setfit import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, f1_score

def compute_metrics(y_pred, y_test):
    y_pred = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    y_test = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1
    }
    
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(device)

# Prepare dataset
df = pd.read_csv("dataset\\manual_dataset.csv")
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(df_train[['text', 'label']])
val_dataset = Dataset.from_pandas(df_val[['text', 'label']])

# Declare model & trainer
model = SetFitModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model.to(device)
args = TrainingArguments(output_dir="./train_results", num_epochs=2, num_iterations=20, logging_steps=10, logging_dir="./logs",)
trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset, metric=compute_metrics)

trainer.train()
print(trainer.evaluate(dataset=val_dataset))

# Save the model
output_dir = "./train_results/trained_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
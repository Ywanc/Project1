from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
import pandas as pd

df = pd.read_csv("dataset\\dataset.csv")
dataset = Dataset.from_pandas(df)

model = SetFitModel.from_pretrained("sentence-transformers/all-distilbert-base-v2")

trainer = SetFitTrainer(model, )
from transformers import DistilBertTokenizerFast, DistilBertConfig
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn import preprocessing
import pandas as pd
import numpy as np


def inference(txt,PATH):
  tokenizer = DistilBertTokenizerFast.from_pretrained(PATH)
  model = DistilBertForSequenceClassification.from_pretrained(PATH)
  inputs = tokenizer(txt, return_tensors="pt")
  outputs = model(**inputs)
  preds = np.argmax(outputs[0][0].detach().numpy(), axis=-1)
  return preds


if __name__ == "__main__":
    PATH = '/content/drive/MyDrive/personal projects/Github_projects/results/checkpoint-30000'
    text = 'I really do not understand why this happened, I thought the situation had normalized'
    res = inference(text,PATH)

    emotions_dict = {'joy':0,'sadness':1,'anger':2,'fear':3,'love':4,'surprice':5}
    key_list = list(emotions_dict.keys())
    val_list = list(emotions_dict.values())
    position = val_list.index(res)
    print(key_list[position])

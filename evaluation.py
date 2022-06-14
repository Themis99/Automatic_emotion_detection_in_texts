from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import numpy as np
import pandas as pd

PATH_TEST = '/content/drive/MyDrive/personal projects/Github_projects/test.txt'
test_set = test_set = pd.read_csv(PATH_TEST, sep=';')
test_set.columns = ['text', 'class']

test_set = test_set.tail(1000)
classes = test_set['class'].value_counts().index.tolist()
labels = list(range(0, 6))

dict_map = {}

for i, j in zip(classes, labels):
    dict_map[i] = j
test_set = test_set.replace({"class": dict_map})

#use preds calculated from the model and load them
precision = precision_score(test_set['class'].tolist(),preds,average='micro')
print('precision: ', precision*100)
recall = recall_score(test_set['class'].tolist(), preds, average='micro')
print('recall: ',recall*100)
acc = accuracy_score(test_set['class'].tolist(), preds)
print('accuracy: ', acc*100)
f1 = f1_score(test_set['class'].tolist(),preds,average='micro')
print('F1 score: ', f1*100)

cm = confusion_matrix(test_set['class'].tolist(), preds)
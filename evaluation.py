from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import numpy as np

with open('/content/drive/MyDrive/personal projects/class_label.pkl', 'rb') as f:
    class_to_labels = pickle.load(f)

PATH_TEST = '/content/drive/MyDrive/personal projects/Github_projects/test.txt'
test_set = test_set = pd.read_csv(PATH_TEST, sep=';')
test_set.columns = ['text', 'class']
test_set = test_set.tail(1000)
test_set = test_set.replace({"class": class_to_labels})
preds = pd.read_csv('/content/drive/MyDrive/personal projects/predictions.csv').drop(['Unnamed: 0'],axis=1)

precision = precision_score(test_set['class'].tolist(),preds,average='micro')
print('precision: ', precision*100)
recall = recall_score(test_set['class'].tolist(), preds, average='micro')
print('recall: ',recall*100)
acc = accuracy_score(test_set['class'].tolist(), preds)
print('accuracy: ', acc*100)
f1 = f1_score(test_set['class'].tolist(),preds,average='micro')
print('F1 score: ', f1*100)

cm = confusion_matrix(test_set['class'].tolist(), preds)

fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(cm,annot=True,linewidths=.5,cmap="crest",xticklabels = class_to_labels.keys(),yticklabels = class_to_labels.keys(),fmt = "d")


from transformers import DistilBertTokenizerFast, DistilBertConfig
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
import torch
import wandb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


class QuestiionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':

    # import data
    PATH_TRAIN = '/content/drive/MyDrive/personal projects/Github_projects/train.txt'
    train_set = train_data = pd.read_csv(PATH_TRAIN, sep=';')
    train_set.columns = ['text', 'class']

    PATH_TEST = '/content/drive/MyDrive/personal projects/Github_projects/test.txt'
    test_set = test_set = pd.read_csv(PATH_TEST, sep=';')
    test_set.columns = ['text', 'class']

    train_set = train_set.tail(6000)
    test_set = test_set.tail(1000)

    classes = train_set['class'].value_counts().index.tolist()
    labels = list(range(0, 6))

    dict_map = {}

    for i, j in zip(classes, labels):
        dict_map[i] = j

    train_set = train_set.replace({"class": dict_map})
    test_set = test_set.replace({"class": dict_map})

    wandb.login()

    # prepare dataset
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_set['text'].tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(test_set['text'].tolist(), truncation=True, padding=True)

    train_dataset = QuestiionDataset(train_encodings, train_set['class'].tolist())
    val_dataset = QuestiionDataset(val_encodings, test_set['class'].tolist())

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(torch_device)
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
    config.num_labels = 6
    model = DistilBertForSequenceClassification(config)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        save_steps=500,
        save_total_limit=5,
        evaluation_strategy='steps',
        eval_steps=100,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    trainer.evaluate(val_dataset)

    predictions = trainer.predict(val_dataset)

    preds = np.argmax(predictions.predictions, axis=-1)

    wandb.finish()
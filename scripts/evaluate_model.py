# scripts/evaluate_model.py

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import classification_report
import os

def evaluate_model():
    model = BertForSequenceClassification.from_pretrained('models')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = torch.load('data/processed/test_dataset.pt')

    training_args = TrainingArguments(
        output_dir='models',
        per_device_eval_batch_size=64,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
    )

    predictions, labels, _ = trainer.predict(test_dataset)
    preds = np.argmax(predictions, axis=1)

    report = classification_report(labels, preds, output_dict=True)
    print(classification_report(labels, preds))

    # Return metrics for Airflow to log
    return report

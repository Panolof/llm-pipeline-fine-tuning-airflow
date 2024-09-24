# scripts/train_model.py

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os

def train_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = torch.load('data/processed/train_dataset.pt')
    val_dataset = torch.load('data/processed/validation_dataset.pt')

    training_args = TrainingArguments(
        output_dir='models',
        num_train_epochs=1,  # Adjust as needed
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='logs',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

# scripts/preprocess_data.py

from transformers import BertTokenizer
from datasets import load_from_disk
import torch
import os

def preprocess_data():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = load_from_disk('data/raw')
    
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    for split in ['train', 'validation', 'test']:
        ds = dataset[split].map(tokenize, batched=True)
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        torch.save(ds, f'data/processed/{split}_dataset.pt')

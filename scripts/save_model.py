# scripts/save_model.py

from transformers import BertForSequenceClassification, BertTokenizer
import os

def save_model():
    model = BertForSequenceClassification.from_pretrained('models')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    os.makedirs('models/saved_model', exist_ok=True)
    model.save_pretrained('models/saved_model')
    tokenizer.save_pretrained('models/saved_model')

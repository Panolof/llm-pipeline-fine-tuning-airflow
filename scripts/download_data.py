# scripts/download_data.py

from datasets import load_dataset
import os

def download_data():
    dataset = load_dataset('dair-ai/emotion')
    os.makedirs('data/raw', exist_ok=True)
    dataset.save_to_disk('data/raw')

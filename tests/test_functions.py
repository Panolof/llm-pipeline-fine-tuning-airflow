# tests/test_functions.py

import unittest
from scripts.download_data import download_data
from scripts.preprocess_data import preprocess_data
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model
from datasets import load_dataset
import os

class TestEmotionDetectionPipeline(unittest.TestCase):

    def test_download_data(self):
        # Ensure data directory is clean
        if os.path.exists('data/raw'):
            os.system('rm -rf data/raw')
        os.makedirs('data/raw', exist_ok=True)
        
        download_data()
        self.assertTrue(os.path.exists('data/raw/dataset.arrow'))

    def test_preprocess_data(self):
        preprocess_data()
        self.assertTrue(os.path.exists('data/processed/train_dataset.pt'))
        self.assertTrue(os.path.exists('data/processed/val_dataset.pt'))
        self.assertTrue(os.path.exists('data/processed/test_dataset.pt'))

    def test_train_model(self):
        train_model()
        self.assertTrue(os.path.exists('models/checkpoint-500'))

    def test_evaluate_model(self):
        metrics = evaluate_model()
        self.assertIn('eval_accuracy', metrics)
        self.assertIn('eval_loss', metrics)

if __name__ == '__main__':
    unittest.main()

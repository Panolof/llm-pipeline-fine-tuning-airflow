# Transformer Model Fine-Tuning with Apache Airflow

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Airflow DAG Details](#airflow-dag-details)
- [Dataset](#dataset)
- [Model Fine-Tuning](#model-fine-tuning)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

This project demonstrates advanced use of **Apache Airflow** to orchestrate a machine learning pipeline for fine-tuning a Transformer model using **Hugging Face**. We focus on an innovative use case: fine-tuning a Transformer to detect emotions in text data from the **Emotion Dataset**. This pipeline showcases deep competency in Airflow, including custom operators, task dependencies, and integration with external libraries.

---

## Project Overview

The goal is to build an end-to-end automated pipeline that:

1. **Ingests and Preprocesses Data**: Downloads the Emotion Dataset and prepares it for training.
2. **Fine-Tunes a Transformer Model**: Uses Hugging Face's `bert-base-uncased` model for emotion classification.
3. **Evaluates the Model**: Assesses performance using metrics like accuracy and F1-score.
4. **Saves the Model**: Stores the fine-tuned model locally or on cloud storage.
5. **Notifies Upon Completion**: Sends an email or Slack notification when the pipeline finishes.

By integrating these steps into an Airflow DAG, we demonstrate how to manage complex workflows, ensure data consistency, and automate machine learning tasks in a production-like environment.

---

## Airflow DAG Details

The Airflow DAG (`dags/emotion_detection_dag.py`) orchestrates the following tasks:

1. **Start**: `DummyOperator` signaling the DAG's initiation.
2. **Data Download**: `PythonOperator` that downloads the Emotion Dataset.
3. **Data Preprocessing**: Cleans and tokenizes text data, encoding labels.
4. **Model Fine-Tuning**: Fine-tunes `bert-base-uncased` for emotion detection.
5. **Model Evaluation**: Evaluates the model using the test set.
6. **Model Saving**: Saves the trained model and tokenizer.
7. **Notification**: `EmailOperator` or `SlackAPIPostOperator` sends completion alerts.
8. **End**: `DummyOperator` signaling the DAG's completion.

**Advanced Airflow Features Demonstrated**:

- **Custom Operators**: Specialized tasks for model training and evaluation.
- **XComs**: Pass data between tasks securely.
- **Error Handling**: Retries and alerting on failures.
- **Dynamic Task Mapping**: For scalable and parallel data processing.
- **Integration with External Libraries**: Seamless use of Hugging Face within Airflow tasks.

---

## Dataset

We utilize the **Emotion Dataset**, containing 20,000 English Twitter messages labeled with six emotions: anger, fear, joy, love, sadness, and surprise.

- **Source**: [Emotion Dataset on Hugging Face Datasets](https://huggingface.co/datasets/dair-ai/emotion)
- **Purpose**: Enables multi-class emotion detection, providing a nuanced understanding beyond binary sentiment analysis.

---

## Model Fine-Tuning

We fine-tune the `bert-base-uncased` Transformer model for emotion classification. Below is a step-by-step guide:

### Step 1: Data Download and Preparation

- **Download Dataset**:

  ```python
  from datasets import load_dataset
  dataset = load_dataset('dair-ai/emotion')
  ```

- **Split Dataset**:

  ```python
  train_dataset = dataset['train']
  val_dataset = dataset['validation']
  test_dataset = dataset['test']
  ```

### Step 2: Data Preprocessing

- **Initialize Tokenizer**:

  ```python
  from transformers import BertTokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  ```

- **Tokenize Data**:

  ```python
  def tokenize(batch):
      return tokenizer(batch['text'], padding=True, truncation=True)
  train_dataset = train_dataset.map(tokenize, batched=True)
  val_dataset = val_dataset.map(tokenize, batched=True)
  test_dataset = test_dataset.map(tokenize, batched=True)
  ```

- **Encode Labels**:

  ```python
  label_encoder = {label: i for i, label in enumerate(train_dataset.features['label'].names)}
  ```

### Step 3: Model Setup

- **Load Pre-trained Model**:

  ```python
  from transformers import BertForSequenceClassification
  model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
  ```

### Step 4: Training

- **Define Training Arguments**:

  ```python
  from transformers import TrainingArguments
  training_args = TrainingArguments(
      output_dir='./results',
      num_train_epochs=3,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=64,
      evaluation_strategy='epoch',
      save_total_limit=2,
      load_best_model_at_end=True,
  )
  ```

- **Initialize Trainer**:

  ```python
  from transformers import Trainer
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      tokenizer=tokenizer,
  )
  ```

- **Train Model**:

  ```python
  trainer.train()
  ```

### Step 5: Evaluation

- **Evaluate Model**:

  ```python
  metrics = trainer.evaluate(eval_dataset=test_dataset)
  print(metrics)
  ```

- **Detailed Metrics**:

  ```python
  import numpy as np
  from sklearn.metrics import classification_report

  predictions, labels, _ = trainer.predict(test_dataset)
  preds = np.argmax(predictions, axis=1)
  report = classification_report(labels, preds, target_names=label_encoder.keys())
  print(report)
  ```

### Step 6: Saving the Model

- **Save Locally**:

  ```python
  model.save_pretrained('./saved_model')
  tokenizer.save_pretrained('./saved_model')
  ```

- **Save to Cloud (Optional)**:

  Upload to AWS S3 or GCP Storage using appropriate SDKs.

### Step 7: Integration into Airflow

Each step is encapsulated in scripts within the `scripts/` directory and executed via Airflow tasks, ensuring modularity and ease of maintenance.

---



























## Project Structure

```
transformer_airflow_project/
├── dags/
│   └── emotion_detection_dag.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── saved_model/
├── scripts/
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── save_model.py
├── tests/
│   └── test_pipeline.py
├── logs/
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Prerequisites

- **Python 3.7+**
- **Apache Airflow 2.x**
- **Hugging Face Transformers**
- **Datasets Library**
- **PyTorch**
- **Slack API Token** (if using Slack notifications)
- **Git**

---

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your_username/transformer_airflow_project.git
   cd transformer_airflow_project
   ```

2. **Set Up Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Initialize Airflow**:

   ```bash
   export AIRFLOW_HOME=~/airflow
   airflow db init
   airflow users create \
       --username admin \
       --firstname YOUR_FIRST_NAME \
       --lastname YOUR_LAST_NAME \
       --role Admin \
       --email your_email@example.com
   ```

4. **Configure Connections**:

   - **Slack**: Add a Slack connection in Airflow with your API token.
   - **AWS/GCP**: Set up connections if saving models to cloud storage.

5. **Install Airflow Providers**:

   ```bash
   pip install apache-airflow-providers-slack
   pip install apache-airflow-providers-http
   ```

---

## Usage

1. **Start Airflow Services**:

   ```bash
   airflow scheduler
   airflow webserver
   ```

2. **Trigger the DAG**:

   - **Via UI**: Navigate to `http://localhost:8080`, find `emotion_detection_dag`, and click the trigger button.
   - **Via CLI**:

     ```bash
     airflow dags trigger emotion_detection_dag
     ```

3. **Monitor Pipeline**:

   - Use the Airflow UI to monitor task progress and logs.
   - Check `logs/` directory for detailed logs.

4. **Review Results**:

   - Evaluated metrics are printed in logs and can be accessed in the Airflow UI.
   - The fine-tuned model is saved in `models/saved_model/`.

---

## Testing

Run unit tests to ensure each component functions correctly.

```bash
python -m unittest discover tests
```

---

## Future Work

- **Data Version Control (DVC)**: Implement DVC to track data and model changes.
- **Hyperparameter Optimization**: Use **Optuna** or **Ray Tune** for automated hyperparameter tuning.
- **Model Serving**: Deploy the model using **TensorFlow Serving** or **TorchServe**.
- **Kubernetes Executor**: Scale Airflow tasks with Kubernetes for distributed execution.
- **CI/CD Integration**: Set up **GitHub Actions** or **Jenkins** for continuous integration and deployment.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

# Potential Future Works

- **Data Augmentation**: Implement techniques to enhance the dataset and improve model robustness.
- **Multi-Language Support**: Extend the pipeline to handle datasets in multiple languages.
- **Automated Retraining**: Set up Airflow sensors to detect new data and trigger retraining.
- **Monitoring and Logging**: Integrate tools like **Prometheus** and **Grafana** for real-time monitoring.
- **Security Enhancements**: Implement authentication and authorization mechanisms for model access.

---

Feel free to explore these ideas and contribute to the project's growth!

---

# Examples of Text Data for the Flow

We use the **Emotion Dataset**, but you can experiment with other text datasets:

- **Amazon Reviews**: For sentiment analysis or product feature extraction.
- **Twitter Sentiment Analysis**: Real-time sentiment tracking on specific hashtags.
- **News Articles**: Topic modeling or fake news detection.
- **Customer Support Tickets**: Intent classification to route tickets.

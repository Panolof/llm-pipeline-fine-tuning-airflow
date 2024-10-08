# Transformer Model Fine-Tuning with Apache Airflow

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Airflow DAG Details](#airflow-dag-details)
- [Dataset](#dataset)
- [Model Fine-Tuning](#model-fine-tuning)
- [Inference](#inference)
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
5. **Performs Inference**: Uses the fine-tuned model to predict emotions on new text data.
6. **Versions Data with DVC**: Automatically versions processed data using Data Version Control (DVC).
7. **Notifies Upon Completion**: Sends an email or Slack notification when the pipeline finishes.

By integrating these steps into an Airflow DAG, we demonstrate how to manage complex workflows, ensure data consistency, automate machine learning tasks, and maintain data versioning in a production-like environment.

---

## Airflow DAG Details

The Airflow DAG (`dags/emotion_detection_dag.py`) orchestrates the following tasks:

1. **Start**: `DummyOperator` signaling the DAG's initiation.
2. **Data Download**: `PythonOperator` that downloads the Emotion Dataset.
3. **Data Preprocessing**: Cleans and tokenizes text data, encoding labels.
4. **Data Versioning**: `PythonOperator` that versions the processed data using DVC.
5. **Model Fine-Tuning**: Fine-tunes `bert-base-uncased` for emotion detection.
6. **Model Evaluation**: Evaluates the model using the test set.
7. **Model Saving**: Saves the trained model and tokenizer.
8. **Notification**: `EmailOperator` or `SlackAPIPostOperator` sends completion alerts.
9. **End**: `DummyOperator` signaling the DAG's completion.

**Advanced Airflow Features Demonstrated**:

- **Custom Operators**: Specialized tasks for model training, evaluation, and data versioning.
- **XComs**: Pass data between tasks securely.
- **Error Handling**: Retries and alerting on failures.
- **Dynamic Task Mapping**: For scalable and parallel data processing.
- **Integration with External Libraries**: Seamless use of Hugging Face within Airflow tasks.
- **Data Versioning with DVC**: Ensures data consistency and reproducibility.

---

## Dataset

We utilize the **Emotion Dataset**, containing 20,000 English Twitter messages labeled with six emotions: anger, fear, joy, love, sadness, and surprise.

- **Source**: [Emotion Dataset on Hugging Face Datasets](https://huggingface.co/datasets/dair-ai/emotion)
- **Purpose**: Enables multi-class emotion detection, providing a nuanced understanding beyond binary sentiment analysis.

---

## Model Fine-Tuning

We fine-tune the `bert-base-uncased` Transformer model for emotion classification. Each step is encapsulated in scripts within the `scripts/` directory and executed via Airflow tasks, ensuring modularity and ease of maintenance.

---

## Inference

Once the model has been fine-tuned and saved, you can use it to predict emotions on new text data using the `scripts/example_data.py` script.

**Running Inference**:

1. **Ensure the Fine-Tuned Model is Saved**:

   Make sure the model has been trained and saved in the `models/saved_model/` directory by running the Airflow pipeline.

2. **Run the Inference Script**:

   ```bash
   python scripts/example_data.py
   ```

   This script will load the fine-tuned model and tokenizer and perform emotion prediction on a set of sample sentences.

**Sample Output**:

```
Text: I'm so happy to hear from you!
Predicted Emotion: joy

Text: This is the worst day ever.
Predicted Emotion: sadness

Text: I can't wait to see you again.
Predicted Emotion: love

Text: I'm feeling very anxious about the meeting.
Predicted Emotion: fear

Text: You did an amazing job!
Predicted Emotion: joy

Text: I'm utterly disappointed.
Predicted Emotion: sadness
```

**Custom Input**:

You can modify the `sample_texts` list in `scripts/example_data.py` to include your own sentences for emotion prediction.

---

## Project Structure

```
transformer_airflow_project/
├── dags/
│   ├── emotion_detection_dag.py
│   └── example_dag.py          # Added example DAG
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
│   ├── save_model.py
│   ├── example_data.py         # Added inference script
│   └── data_versioning.py      # Added data versioning script
├── tests/
│   ├── __init__.py
│   └── test_functions.py
├── logs/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

---

## Prerequisites

- **Python 3.7+**
- **Apache Airflow 2.x**
- **Hugging Face Transformers 4.21.0**
- **Datasets Library 2.3.2**
- **PyTorch 1.11.0**
- **Scikit-learn 0.24.2**
- **Pandas 1.3.0**
- **NumPy 1.21.0**
- **Slack API Token** (if using Slack notifications)
- **DVC 2.7.3** (for data versioning)
- **Git**
- **Docker** (if using Docker)
- **GitHub Actions** (for CI/CD)

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
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Initialize Airflow**:

   ```bash
   export AIRFLOW_HOME=$(pwd)/airflow
   airflow db init
   airflow users create \
       --username admin \
       --firstname YOUR_FIRST_NAME \
       --lastname YOUR_LAST_NAME \
       --role Admin \
       --email your_email@example.com
   ```

4. **Set Up Airflow Configuration**:

   - Ensure the `dags_folder` in `airflow.cfg` points to the `dags/` directory in your project.
   - Set any required environment variables or connections.

5. **Configure Connections and Variables**:

   - **Slack**: Add a Slack connection in Airflow with your API token.
   - **AWS/GCP**: Set up connections if saving models to cloud storage.
   - **Environment Variables**: Set any required environment variables.

6. **Install Airflow Providers**:

   ```bash
   pip install apache-airflow-providers-slack
   pip install apache-airflow-providers-email
   ```

7. **Initialize DVC**:

   ```bash
   dvc init
   dvc remote add -d s3remote s3://your-bucket-name/path
   dvc remote modify s3remote endpointurl https://s3.amazonaws.com
   ```
   Replace your-bucket-name/path with your actual S3 bucket and path.

---

## Usage

### 1. Start Airflow Services

Start the Airflow scheduler and webserver in separate terminal windows or as background processes:

```bash
airflow scheduler &
airflow webserver -p 8080 &
```

### 2. Access Airflow UI

Open your web browser and navigate to [http://localhost:8080](http://localhost:8080) to access the Airflow UI.

### 3. Trigger the DAGs

#### a. **Emotion Detection DAG**

- **Via UI**:
  - Locate the `emotion_detection_dag` in the DAGs list.
  - Click the toggle to activate the DAG if it's not already active.
  - Click the **Trigger DAG** button to start the pipeline.

- **Via CLI**:

  ```bash
  airflow dags trigger emotion_detection_dag
  ```

#### b. **Example DAG**

The `example_dag.py` is a simple DAG included for demonstration and testing purposes.

- **Purpose**: To verify that your Airflow setup is functioning correctly by running a minimal DAG with no real tasks.

- **Via UI**:
  - Locate the `example_dag` in the DAGs list.
  - Click the toggle to activate the DAG if it's not already active.
  - Click the **Trigger DAG** button to run the example DAG.

- **Via CLI**:

  ```bash
  airflow dags trigger example_dag
  ```

**Note**: The `example_dag` contains only `DummyOperator` tasks (`start` and `end`) and serves as a template or a sanity check for your Airflow installation.

### 4. Monitor Pipeline

- **Airflow UI**:
  - Use the **Graph View** or **Tree View** to monitor task progress.
  - Click on individual tasks to view logs and troubleshoot if necessary.

- **Logs Directory**:
  - Check the `logs/` directory for detailed logs of each task execution.

### 5. Review Results

- **Emotion Detection Pipeline**:
  - The fine-tuned model is saved in `models/saved_model/`.
  - Evaluation metrics are available in the task logs and can be accessed via the Airflow UI.

- **Example DAG**:
  - Since it uses `DummyOperator`, it won't produce any outputs but confirms that Airflow is operational.

### 6. Run Inference

After training and saving the model, perform inference using the `example_data.py` script:

```bash
python scripts/example_data.py
```

This script will output predicted emotions for the sample sentences provided.

---

## Testing

Run unit tests to ensure each component functions correctly.

```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Run unit tests
python -m unittest discover tests
```

**Note**: Ensure that the Airflow services are not running during testing to prevent conflicts, especially if using local SQLite databases.

---

## Future Work

- **Data Version Control (DVC)**: Implement DVC to track data and model changes.
- **Hyperparameter Optimization**: Use **Optuna** or **Ray Tune** for automated hyperparameter tuning.
- **Model Serving**: Deploy the model using **TensorFlow Serving** or **TorchServe**.
- **Kubernetes Executor**: Scale Airflow tasks with Kubernetes for distributed execution.
- **CI/CD Integration**: Set up **GitHub Actions** or **Jenkins** for continuous integration and deployment.
- **Dockerization**: Containerize the application for easier deployment and scalability.
- **Model Registry**: Use tools like **MLflow** for model tracking and management.
- **Advanced Monitoring**: Integrate **Prometheus** and **Grafana** for real-time monitoring and alerting.
- **Security Enhancements**: Implement authentication and authorization mechanisms for model access.
- **Automated Data Validation**: Add tasks to validate data quality before processing.
- **Scheduled Model Retraining**: Automate periodic retraining of models with fresh data.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Potential Future Works

- **Data Augmentation**: Implement techniques to enhance the dataset and improve model robustness.
- **Multi-Language Support**: Extend the pipeline to handle datasets in multiple languages.
- **Automated Retraining**: Set up Airflow sensors to detect new data and trigger retraining.
- **Monitoring and Logging**: Integrate tools like **Prometheus** and **Grafana** for real-time monitoring.
- **Security Enhancements**: Implement authentication and authorization mechanisms for model access.

---

Feel free to explore these ideas and contribute to the project's growth!

---

## Examples of Text Data for the Flow

We use the **Emotion Dataset**, but you can experiment with other text datasets:

- **Amazon Reviews**: For sentiment analysis or product feature extraction.
- **Twitter Sentiment Analysis**: Real-time sentiment tracking on specific hashtags.
- **News Articles**: Topic modeling or fake news detection.
- **Customer Support Tickets**: Intent classification to route tickets.

---

**Note**: Remember to update the `CONTRIBUTING.md` and `LICENSE` files if any changes have been made.
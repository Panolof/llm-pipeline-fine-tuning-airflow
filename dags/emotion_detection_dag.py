# dags/emotion_detection_dag.py

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.email_operator import EmailOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os

# Import your custom scripts
from scripts.download_data import download_data
from scripts.preprocess_data import preprocess_data
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model
from scripts.save_model import save_model

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),  # Adjust as needed
    'depends_on_past': False,
    'email': ['your_email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'emotion_detection_dag',
    default_args=default_args,
    description='An Airflow DAG to fine-tune a Transformer model for emotion detection',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    start = DummyOperator(
        task_id='start',
    )

    task_download_data = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
    )

    task_preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    task_train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    task_evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    task_save_model = PythonOperator(
        task_id='save_model',
        python_callable=save_model,
    )

    notify = EmailOperator(
        task_id='send_email',
        to='your_email@example.com',
        subject='Airflow Notification: Emotion Detection Pipeline Completed',
        html_content='The emotion detection pipeline has successfully completed.',
    )

    end = DummyOperator(
        task_id='end',
    )

    # Define task dependencies
    start >> task_download_data >> task_preprocess_data >> task_train_model
    task_train_model >> task_evaluate_model >> task_save_model >> notify >> end

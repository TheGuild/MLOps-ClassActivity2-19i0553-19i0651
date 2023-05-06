from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Define default arguments for the DAG
default_args = {
    'owner': 'Navira-Adeen',
    'depends_on_past': False,
    'start_date': datetime(2023, 5, 6),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Instantiate the DAG
dag = DAG(
    'auto_arima_pipeline',
    default_args=default_args,
    description='End-to-end pipeline for AutoARIMA model',
    schedule=timedelta(days=1),
)

# Define the data preprocessing task
preprocess_data = BashOperator(
    task_id='preprocess_data',
    bash_command='python3 data_preprocessing.py',
    dag=dag,
)

# Define the model training task
train_model = BashOperator(
    task_id='train_model',
    bash_command='python3 salesprediction.py',
    dag=dag,
)

# Define the model evaluation task
evaluate_model = BashOperator(
    task_id='evaluate_model',
    bash_command='python3 evaluate.py',
    dag=dag,
)

# Define the model deployment task
deploy_model = BashOperator(
    task_id='deploy_model',
    bash_command='python3 deploy.py',
    dag=dag,
)

# Define task dependencies
preprocess_data >> train_model >> evaluate_model >> deploy_model

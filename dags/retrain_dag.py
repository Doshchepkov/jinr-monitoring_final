"""
Airflow DAG для периодического дообучения модели
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'jinr',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'jinr_model_retrain',
    default_args=default_args,
    description='Периодическое дообучение модели на новых положительных примерах',
    schedule_interval='0 2 * * 0',  # Каждое воскресенье в 2:00
    catchup=False,
    tags=['jinr', 'retrain'],
)

# Пути
PROJECT_DIR = '/opt/jinr-monitoring'
BASE_DATA = f'{PROJECT_DIR}/datasets/merged_dataset2.csv'
MODEL_PATH = f'{PROJECT_DIR}/models/final_xgb.pkl'
RETRAIN_SCRIPT = f'{PROJECT_DIR}/retrain.py'

# Задача: проверить наличие новых эпизодов
check_buffer = BashOperator(
    task_id='check_buffer',
    bash_command=f'python -c "import sys; sys.path.insert(0, \\"{PROJECT_DIR}/src\\"); from src.buffer import PositiveBuffer; b = PositiveBuffer(); print(f\\"Buffer has {b.unused_size()} new events\\")"',
    dag=dag,
)

# Задача: дообучение модели
retrain_model = BashOperator(
    task_id='retrain_model',
    bash_command=f'cd {PROJECT_DIR} && python {RETRAIN_SCRIPT} --base-data {BASE_DATA} --original-model {MODEL_PATH} --new-model {MODEL_PATH}.new',
    dag=dag,
)

# Задача: замена модели (если новая лучше)
replace_model = BashOperator(
    task_id='replace_model',
    bash_command=f'cd {PROJECT_DIR} && mv {MODEL_PATH}.new {MODEL_PATH}',
    dag=dag,
)

# Задача: уведомление (опционально)
notify = BashOperator(
    task_id='notify',
    bash_command='echo "Model retraining completed"',
    dag=dag,
)

# Порядок выполнения
check_buffer >> retrain_model >> replace_model >> notify
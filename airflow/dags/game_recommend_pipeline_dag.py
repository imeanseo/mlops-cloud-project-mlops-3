from datetime import timedelta
import random

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'game_recommend_pipeline_dag_v7',
    default_args=default_args,
    description='Daily Game Recommend pipeline (data prepare + training + recommend_all + save to MySQL + Slack)',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops']
) as dag:

    # Step 1: 데이터 수집 + 전처리
    step1_prepare = DockerOperator(
        task_id='data_prepare',
        image='moongs95/third-party-mlops:v7',
        api_version='auto',
        auto_remove=True,
        command='python data-prepare/main.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops',
        environment={
            'RAWG_API_KEY': '{{ var.value.RAWG_API_KEY }}'
        }
    )

    # Step 2: 모델 학습 + W&B 로깅
    step2_train = DockerOperator(
        task_id='model_train',
        image='moongs95/third-party-mlops:v7',
        api_version='auto',
        auto_remove=True,
        command='python mlops/src/main.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops',
        environment={
            'WANDB_API_KEY': '{{ var.value.WANDB_API_KEY }}',
            'AWS_ACCESS_KEY_ID': '{{ var.value.AWS_ACCESS_KEY_ID }}',
            'AWS_SECRET_ACCESS_KEY': '{{ var.value.AWS_SECRET_ACCESS_KEY }}',
            'S3_BUCKET_NAME': '{{ var.value.S3_BUCKET_NAME }}',
        }
    )

    # Step 3: Recommend All
    step3_recommend_all = DockerOperator(
        task_id='recommend_all',
        image='moongs95/third-party-mlops:v7',
        api_version='auto',
        auto_remove=True,
        command='python mlops/src/main.py recommend_all',
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops',
        environment={
            'WANDB_API_KEY': '{{ var.value.WANDB_API_KEY }}',
            'AWS_ACCESS_KEY_ID': '{{ var.value.AWS_ACCESS_KEY_ID }}',
            'AWS_SECRET_ACCESS_KEY': '{{ var.value.AWS_SECRET_ACCESS_KEY }}',
            'S3_BUCKET_NAME': '{{ var.value.S3_BUCKET_NAME }}',
        }
    )

    # Step 4: CSV를 MySQL에 저장
    step4_save_to_db = DockerOperator(
        task_id='save_to_mysql',
        image='moongs95/third-party-mlops:v7',
        api_version='auto',
        auto_remove=True,
        command='python mlops/src/save_to_db.py',
        docker_url='unix://var/run/docker.sock',
        network_mode='mlops',
        environment={
            'MYSQL_HOST': 'game-mlops-db',
            'MYSQL_PORT': '3306',
            'MYSQL_USER': 'root',
            'MYSQL_PASSWORD': 'root',
            'MYSQL_DB': 'mlops',
            'WANDB_API_KEY': '{{ var.value.WANDB_API_KEY }}',
            'AWS_ACCESS_KEY_ID': '{{ var.value.AWS_ACCESS_KEY_ID }}',
            'AWS_SECRET_ACCESS_KEY': '{{ var.value.AWS_SECRET_ACCESS_KEY }}',
            'S3_BUCKET_NAME': '{{ var.value.S3_BUCKET_NAME }}',
        }
    )

    # Step 5: Slack Notification (학습 + 저장 완료)
    slack_notify = SlackWebhookOperator(
        task_id='slack_notify',
        slack_webhook_conn_id='slack_webhook',
        message="🎉 MLOps DAG v7 - 전체 파이프라인이 완료되었습니다! 모델과 DB가 최신 상태로 업데이트 되었습니다.",
        username="airflow_bot"
    )

    # DAG 실행 순서
    step1_prepare >> step2_train >> step3_recommend_all >> step4_save_to_db >> slack_notify

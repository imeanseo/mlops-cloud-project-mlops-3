from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.models import Variable
from docker.types import Mount
from datetime import timedelta
import pendulum
import os

# 환경 변수
RAWG_API_KEY = Variable.get("RAWG_API_KEY", default_var=None)
WANDB_API_KEY = Variable.get("WANDB_API_KEY", default_var=None)

if RAWG_API_KEY is None:
    RAWG_API_KEY = os.environ.get("AIRFLOW_VAR_RAWG_API_KEY")
if WANDB_API_KEY is None:
    WANDB_API_KEY = os.environ.get("AIRFLOW_VAR_WANDB_API_KEY")

TZ = pendulum.timezone(os.environ.get("TZ", "Asia/Seoul"))
SLACK_CONN_ID = "slack_default"

# 공유 볼륨 경로
SHARED_MODELS_PATH = "/home/ailee/airflow_imeanseo_ml3/ml3_airflow/opt/shared/models"

default_args = {
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=60),
    "depends_on_past": False,
}

with DAG(
    dag_id="game_recommend_full_pipeline",
    start_date=pendulum.datetime(2025, 1, 1, tz=TZ),
    schedule=None,
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["docker", "mlops", "game-recommend"],
) as dag:

    # 1. 데이터 수집 + 전처리 (마운트 불필요)
    data_prepare_task = DockerOperator(
        task_id="data_prepare",
        image="moongs95/third-party-mlops:v5",
        command="python data-prepare/main.py --limit 40",
        docker_url="unix://var/run/docker.sock",  # 변경!
        auto_remove=True,
        network_mode="bridge",
        mount_tmp_dir=False,
        environment={
            "RAWG_API_KEY": RAWG_API_KEY,
            "WANDB_API_KEY": WANDB_API_KEY,
            "TZ": "Asia/Seoul"
        },
        # mounts 없음 - 맞습니다!
    )

    # 2. 모델 학습 (마운트 필수!)
    train_model_task = DockerOperator(
        task_id="train_model",
        image="moongs95/third-party-mlops:v5",
        command="python mlops/src/main.py train",
        docker_url="unix://var/run/docker.sock",  # 변경!
        auto_remove=True,
        network_mode="bridge",
        mount_tmp_dir=False,
        environment={
            "RAWG_API_KEY": RAWG_API_KEY,
            "WANDB_API_KEY": WANDB_API_KEY,
            "TZ": "Asia/Seoul"
        },
        mounts=[
            Mount(
                source=SHARED_MODELS_PATH,
                target="/opt/mlops/models",
                type="bind"
            )
        ],  # 필수! 모델 저장
    )

    # 3. 추천 생성 (마운트 필수!)
    recommend_task = DockerOperator(
        task_id="recommend",
        image="moongs95/third-party-mlops:v5",
        command="python mlops/src/main.py recommend 12",
        docker_url="unix://var/run/docker.sock",  # 변경!
        auto_remove=True,
        network_mode="bridge",
        mount_tmp_dir=False,
        environment={
            "RAWG_API_KEY": RAWG_API_KEY,
            "WANDB_API_KEY": WANDB_API_KEY,
            "TZ": "Asia/Seoul"
        },
        mounts=[
            Mount(
                source=SHARED_MODELS_PATH,
                target="/opt/mlops/models",
                type="bind"
            )
        ],  # 필수! 모델 로드
    )

    # 4. Slack 성공 알림
    slack_success = SlackWebhookOperator(
        task_id="slack_notify_success",
        slack_webhook_conn_id=SLACK_CONN_ID,
        message="우르르롹끼! 전체 파이프라인 DAG 완료🐵",
        trigger_rule="all_success",
        username="airflow-bot",
        icon_emoji="🐵",
    )

    # Task 순서
    data_prepare_task >> train_model_task >> recommend_task >> slack_success


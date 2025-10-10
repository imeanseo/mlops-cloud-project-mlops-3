import os
import random
from glob import glob

import boto3
import numpy as np



def init_seed():
    np.random.seed(42)
    random.seed(42)


def project_path():
    return os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
        ".."
    )


def model_dir(model_name):
    return os.path.join(
        project_path(),
        "models",
        model_name
    )

def auto_increment_run_suffix(name: str, pad=3):
    suffix = name.split("-")[-1]
    next_suffix = str(int(suffix) + 1).zfill(pad)
    return name.replace(suffix, next_suffix)

def get_s3_client():
    """환경변수 기반 S3 클라이언트 생성"""
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "ap-northeast-2")
    )


def upload_to_s3(local_path, bucket, s3_key):
    s3_client = get_s3_client()
    s3_client.upload_file(local_path, bucket, s3_key)
    print(f"✅ Uploaded {local_path} to s3://{bucket}/{s3_key}")


def download_latest_model_from_s3(bucket_name: str, prefix: str, local_dir: str) -> str:
    """
    S3에서 prefix 경로의 최신 모델(.pkl) 다운로드
    """
    os.makedirs(local_dir, exist_ok=True)
    s3_client = get_s3_client()

    # S3 객체 목록 가져오기
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" not in response:
        raise FileNotFoundError(f"S3에 {prefix} 경로가 없거나 파일이 없습니다.")

    # 최신 파일 선택 (이름 기준 정렬)
    pkl_files = [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError("S3에 .pkl 파일이 없습니다.")

    pkl_files.sort()
    latest_file = pkl_files[-1]

    local_path = os.path.join(local_dir, os.path.basename(latest_file))
    s3_client.download_file(bucket_name, latest_file, local_path)
    print(f"✅ S3에서 최신 모델 다운로드 완료: {local_path}")

    return local_path

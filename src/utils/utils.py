# src/utils/utils.py

import os
import random
import numpy as np
import torch
import boto3
import glob
import pickle
from datetime import datetime


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


# ============================================
# S3 관련 함수들 (수정 버전)
# ============================================

def get_s3_client():
    """환경변수에서 AWS 자격증명을 읽어 S3 클라이언트 생성"""
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "ap-northeast-2")
    )


def upload_file_to_s3(local_path: str, bucket_name: str, s3_key: str):
    """로컬 파일을 S3에 업로드"""
    s3_client = get_s3_client()
    s3_client.upload_file(local_path, bucket_name, s3_key)
    print(f"✅ Uploaded {local_path} to s3://{bucket_name}/{s3_key}")
    return f"s3://{bucket_name}/{s3_key}"


def download_file_from_s3(bucket_name: str, s3_key: str, local_path: str):
    """S3에서 파일 다운로드"""
    s3_client = get_s3_client()
    s3_client.download_file(bucket_name, s3_key, local_path)
    print(f"✅ Downloaded s3://{bucket_name}/{s3_key} to {local_path}")
    return local_path


def get_latest_model_from_s3(bucket_name: str, model_name: str):
    """S3에서 최신 모델 파일 경로 가져오기"""
    s3_client = get_s3_client()
    
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        Prefix=f'models/{model_name}/'
    )
    
    if 'Contents' not in response:
        raise FileNotFoundError(f"S3에 모델 없음: {bucket_name}/models/{model_name}/")
    
    pkl_objects = [obj for obj in response['Contents'] if obj['Key'].endswith('.pkl')]
    
    if not pkl_objects:
        raise FileNotFoundError(f"S3에 .pkl 파일 없음")
    
    latest_obj = max(pkl_objects, key=lambda x: x['LastModified'])
    print(f"📦 Latest model: {latest_obj['Key']} (Modified: {latest_obj['LastModified']})")
    return latest_obj['Key'], latest_obj['LastModified']


def load_model_from_s3(bucket_name: str, model_name: str):
    """S3에서 최신 모델 다운로드 및 로드"""
    s3_key, _ = get_latest_model_from_s3(bucket_name, model_name)
    
    temp_path = f"/tmp/{os.path.basename(s3_key)}"
    download_file_from_s3(bucket_name, s3_key, temp_path)
    
    with open(temp_path, 'rb') as f:
        model_data = pickle.load(f)
    
    os.remove(temp_path)
    print(f"🗑️  Removed temp file: {temp_path}")
    return model_data


def upload_model_to_s3(model_name: str, bucket_name: str, model_type="itemCF"):
    """로컬 모델을 S3에 업로드"""
    print(f"📤 Uploading {model_type} model to S3...")
    
    save_dir = model_dir(model_name)
    pattern = os.path.join(save_dir, f"{model_type}_*.pkl")
    model_files = glob.glob(pattern)
    
    if not model_files:
        raise FileNotFoundError(f"모델 파일 없음: {pattern}")
    
    latest_model_path = max(model_files, key=os.path.getmtime)
    model_filename = os.path.basename(latest_model_path)
    
    print(f"✅ Latest model: {model_filename}")
    
    s3_key = f"models/{model_name}/{model_filename}"
    upload_file_to_s3(latest_model_path, bucket_name, s3_key)
    
    print(f"✅ Model uploaded: {model_filename}")
    return s3_key


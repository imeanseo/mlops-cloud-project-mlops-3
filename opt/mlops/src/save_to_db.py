"""
S3에서 최신 추천 결과 CSV를 읽어 MySQL(Docker)에 저장
"""

import os
import sys

# src 패키지 import 가능하도록 경로 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, Integer, String, Float
from src.utils.utils import get_s3_client  # src/utils/utils.py 사용


def get_latest_csv(bucket_name: str, prefix: str) -> str:
    """S3에서 최신 CSV 파일 Key 반환"""
    s3_client = get_s3_client()
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" not in response:
        raise FileNotFoundError(f"S3 폴더에 파일이 없습니다: s3://{bucket_name}/{prefix}")

    # Key 리스트 중 CSV만 필터링
    files = [obj for obj in response["Contents"] if obj["Key"].endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"CSV 파일이 없습니다: s3://{bucket_name}/{prefix}")

    # 최신 파일 선택 (LastModified 기준)
    latest_file = max(files, key=lambda x: x["LastModified"])["Key"]
    print(f"📥 최신 CSV 선택: {latest_file}")
    return latest_file


def save_recommendations_to_mysql():
    """S3 → MySQL 저장"""
    print("🚀 Starting save_to_mysql...")

    # .env 로드
    load_dotenv()

    bucket_name = os.getenv("S3_BUCKET_NAME", "third-party-game-mlops")
    prefix = "inference_results/"

    # 최신 CSV 파일 Key 가져오기
    s3_client = get_s3_client()
    latest_key = get_latest_csv(bucket_name, prefix)

    # S3에서 CSV 읽기
    obj = s3_client.get_object(Bucket=bucket_name, Key=latest_key)
    df = pd.read_csv(obj["Body"])
    print(f"✅ Loaded {len(df)} rows from {latest_key}")

    # MySQL 연결 정보
    mysql_host = os.getenv("MYSQL_HOST", "game-mlops-db")  # 도커 네트워크 내부 이름
    mysql_port = os.getenv("MYSQL_PORT", "3306")
    mysql_user = os.getenv("MYSQL_USER", "root")
    mysql_password = os.getenv("MYSQL_PASSWORD", "root")
    mysql_database = os.getenv("MYSQL_DATABASE", "mlops")

    DATABASE_URL = f"mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_database}"
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    print(f"🔗 Connecting to MySQL: {mysql_host}:{mysql_port}/{mysql_database}")

    # 테이블 저장
    df.to_sql(
        name="game_recommendations",
        con=engine,
        if_exists="replace",  # 매일 갱신
        index=False,
        dtype={
            "user_id": Integer,
            "game_id": Integer,
            "game_name": String(255),
            "rating": Float,
            "genre": String(100),
        },
        chunksize=1000,
    )

    print(f"✅ Saved to MySQL: {len(df)} rows")


if __name__ == "__main__":
    save_recommendations_to_mysql()

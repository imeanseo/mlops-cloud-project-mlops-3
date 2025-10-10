# src/main.py (원본 유지 + 새 기능 추가)

import os
import sys
from datetime import datetime

# sys.path 경로 설정
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import fire
from dotenv import load_dotenv
import wandb

from src.utils.utils import project_path, auto_increment_run_suffix, upload_model_to_s3, upload_file_to_s3
from src.dataset.games_log import load_games_log
from src.dataset.data_loader import create_user_item_matrix, train_val_split
from src.train.train import train_model
from src.inference.inference import ItemCFInference, run_inference


# 동적으로 경로 계산 (배치 추론용)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MLOPS_DIR = os.path.dirname(SCRIPT_DIR)
OPT_DIR = os.path.dirname(MLOPS_DIR)
DEFAULT_GAME_INFO_PATH = os.path.join(OPT_DIR, 'data-prepare', 'result', 'popular_games.csv')


def get_runs(project_name):
    """WandB runs 가져오기"""
    try:
        api = wandb.Api()
        entity = api.default_entity

        if entity:
            return api.runs(f"{entity}/{project_name}", order="-created_at")
        else:
            return api.runs(project_name, order="-created_at")

    except Exception as e:
        print(f"Error fetching runs: {e}")
        return []


def get_latest_run(project_name):
    """최신 run 이름 가져오기"""
    try:
        runs = get_runs(project_name)
        runs_list = list(runs)

        if not runs_list:
            return None

        import re
        for run in runs_list:
            if re.search(r'-\d+$', run.name):
                return run.name

        return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    """
    모델 학습 (기존 로직)
    """
    # 1. WandB API Key
    load_dotenv(os.path.join(project_path(), ".env"))
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY가 .env 파일에 없습니다.")
    wandb.login(key=wandb_api_key)

    # 2. 데이터 로드
    df = load_games_log("games_log.csv")

    # 3. 유저-아이템 행렬 생성
    user_item_matrix = create_user_item_matrix(df)

    # 4. Train/Validation 분할
    train_matrix, val_matrix = train_val_split(user_item_matrix, val_ratio=0.2, seed=42)

    # 5. WandB run 이름 생성
    project_name = "game_item_cf_recommendation"

    try:
        latest_run = get_latest_run(project_name)
        print(f"Latest run found: {latest_run}")

        if latest_run:
            desired_run_name = auto_increment_run_suffix(latest_run)
            if desired_run_name:
                print(f"Incremented to: {desired_run_name}")
            else:
                print("Failed to increment, using default")
                desired_run_name = f"{project_name}-001"
        else:
            print("No previous runs found, using default")
            desired_run_name = f"{project_name}-001"

    except Exception as e:
        print(f"Error getting run name: {e}")
        desired_run_name = f"{project_name}-001"

    print(f"Final run name: {desired_run_name}")

    # 6. WandB 초기화
    wandb.init(
        project=project_name,
        name=desired_run_name,
        notes="item-based CF recommendation model",
        tags=["itemCF", "recommendation", "games"],
        config={"n_epochs": 10}
    )

    # 7. 모델 학습
    model, recall_history = train_model(
        train_matrix,
        val_matrix,
        n_epochs=10,
        project_name=project_name
    )

    # 8. 특정 유저 추천 (테스트)
    target_user = 10
    recommended_games = model.recommend(target_user, train_matrix, top_k=5)
    print(f"\nUser {target_user} 추천 결과:")
    print(recommended_games)

    # 9. WandB 로그
    wandb.log({"final_recall": recall_history[-1]})
    wandb.finish()

    return model, recall_history


def train():
    """
    학습 + S3 업로드 (CLI 명령)
    """
    print("🚀 Starting training pipeline...")

    # 학습
    model, recall_history = main()
    print("✅ Training completed")

    # S3 업로드
    bucket_name = os.getenv('S3_BUCKET_NAME')
    if bucket_name:
        upload_model_to_s3('itemCF', bucket_name, model_type="itemCF")
        print("🎉 Model uploaded to S3!")
    else:
        print("⚠️ S3_BUCKET_NAME not set, skipping upload")

    return model, recall_history


def recommend(user_id: int, top_k: int = 5):
    """
    단일 사용자 추론 (CLI 명령)
    """
    model_name = "itemCF"
    recommender = ItemCFInference(model_name=model_name)
    games = recommender.recommend(user_id, top_k)
    print(f"user_id={user_id} 추천 결과: {games}")
    return games


def batch_inference(start_user_id: int = 1, end_user_id: int = 100, top_k: int = 12):
    """
    배치 추론 (CLI 명령)
    나중에 inference.py에 구현 후 연결
    """
    print("⚠️ batch_inference 구현 중...")
    # TODO: inference.py의 batch_inference_pipeline() 호출
    pass


# ============================================
# 새로 추가: S3 기반 배치 추론
# ============================================

def recommend_all(s3_bucket: str, model_name: str = 'itemCF', 
                  game_info_path: str = None, output_dir: str = './outputs',
                  s3_prefix: str = 'inference_results'):
    """
    유저 1-100번 추론 + S3 업로드 (새 기능)
    """
    if game_info_path is None:
        game_info_path = DEFAULT_GAME_INFO_PATH
    
    print("\n" + "=" * 60)
    print("🎮 RECOMMEND ALL - 유저 1-100번 게임 추천")
    print("=" * 60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"user_recommendations_1-100_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # 추론 실행
    result_path = run_inference(
        bucket_name=s3_bucket,
        model_name=model_name,
        game_info_path=game_info_path,
        start_user=1,
        end_user=100,
        top_k=5,
        output_path=output_path
    )
    
    # S3 업로드
    print("\n" + "=" * 60)
    print("📤 S3 업로드 시작")
    print("=" * 60)
    s3_key = f"{s3_prefix}/{output_filename}"
    s3_path = upload_file_to_s3(result_path, s3_bucket, s3_key)
    
    print("\n" + "=" * 60)
    print("✅ 전체 작업 완료!")
    print("=" * 60)
    print(f"   로컬 파일: {result_path}")
    print(f"   S3 경로: {s3_path}")
    print("=" * 60 + "\n")


def upload_model(s3_bucket: str, model_name: str = 'itemCF'):
    """
    로컬 모델을 S3에 업로드 (새 기능)
    """
    print("\n" + "=" * 60)
    print("📤 모델 S3 업로드")
    print("=" * 60)
    
    s3_key = upload_model_to_s3(model_name, s3_bucket, "itemCF")
    
    print("\n" + "=" * 60)
    print("✅ 모델 업로드 완료!")
    print("=" * 60)
    print(f"   S3 경로: s3://{s3_bucket}/{s3_key}")
    print("=" * 60 + "\n")


# ============================================
# Main 실행
# ============================================
if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "recommend": recommend,
        "batch_inference": batch_inference,
        "recommend_all": recommend_all,  # 새로 추가
        "upload_model": upload_model      # 새로 추가
    })


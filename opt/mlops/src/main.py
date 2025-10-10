import os
import sys
from datetime import datetime, timezone, timedelta

# sys.path 경로 리스트 나옴
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import fire
from dotenv import load_dotenv
import wandb

from src.utils.utils import project_path, auto_increment_run_suffix, upload_to_s3
from src.dataset.games_log import load_games_log
from src.dataset.data_loader import create_user_item_matrix, train_val_split
from src.train.train import train_model
from src.model.game_item_cf import ItemCF
from src.evaluate.evaluates import recommend_items
from src.inference.inference import ItemCFInference, recommend_all_to_csv

def get_runs(project_name):
    try:
        api = wandb.Api()

        # 현재 로그인된 사용자의 entity 자동 감지
        entity = api.default_entity

        if entity:
            return api.runs(f"{entity}/{project_name}", order="-created_at")
        else:
            # entity 없으면 프로젝트명만으로 시도
            return api.runs(project_name, order="-created_at")

    except Exception as e:
        print(f"Error fetching runs: {e}")
        return []

def get_latest_run(project_name):
    try:
        runs = get_runs(project_name)
        runs_list = list(runs)

        if not runs_list:
            return None

        # 숫자로 끝나는 실제 학습 실행만 찾기
        import re
        for run in runs_list:
            if re.search(r'-\d+$', run.name):
                return run.name

        return None

    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    # ----------------------
    # 1. .env에서 WandB API Key 가져오기
    # ----------------------
    load_dotenv(os.path.join(project_path(), ".env"))
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY가 .env 파일에 없습니다.")
    wandb.login(key=wandb_api_key)

    # ----------------------
    # 2. 데이터 로드
    # ----------------------
    df = load_games_log("games_log.csv")

    # ----------------------
    # 3. 유저-아이템 행렬 생성
    # ----------------------
    user_item_matrix = create_user_item_matrix(df)

    # ----------------------
    # 4. Train/Validation 분할
    # ----------------------
    train_matrix, val_matrix = train_val_split(user_item_matrix, val_ratio=0.2, seed=42)

    # ----------------------
    # 5. 모델 학습 + WandB 로그
    # ----------------------
    project_name = "game_item_cf_recommendation"

    try:
        latest_run = get_latest_run(project_name)
        print(f"Latest run found: {latest_run}")  # 디버깅용

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

    project_name = "game_item_cf_recommendation"

    wandb.init(
        project=project_name,
        name=desired_run_name,
        notes="item-based CF recommendation model",
        tags=["itemCF", "recommendation", "games"],
        config={"n_epochs": 10}
    )


    model, recall_history, model_path = train_model(
        train_matrix,
        val_matrix,
        n_epochs=10,
        project_name=project_name
    )


    # ----------------------
    # 6. 특정 유저 추천
    # ----------------------
    target_user = 10
    recommended_games = model.recommend(target_user, train_matrix, top_k=5)
    print(f"\nUser {target_user} 추천 결과:")
    print(recommended_games)

    # ----------------------
    # 7. WandB 로그
    # ----------------------
    wandb.log({"final_recall": recall_history[-1]})
    wandb.finish()

    # ----------------------
    # 8. 모델을 S3에 업로드
    # ----------------------
    bucket = os.getenv("S3_BUCKET_NAME")
    if bucket:
        s3_key = f"models/itemCF/{os.path.basename(model_path)}"
        upload_to_s3(model_path, bucket, s3_key)
    else:
        print("S3_BUCKET_NAME 환경변수가 없습니다. 업로드 생략.")

def recommend(user_id: int, top_k: int = 5):
    """
    커맨드라인에서 추론 테스트
    """
    model_name = "itemCF"
    recommender = ItemCFInference(model_name=model_name)
    games = recommender.recommend(user_id, top_k)
    print(f"user_id={user_id} 추천 결과: {games}")
    return games

# ----------------------
# 추론 CSV 생성 및 S3 업로드
# ----------------------
def recommend_all(top_k: int = 5):
    """
    1~100번 유저 추천 결과 CSV 생성 + S3 업로드
    """
    user_ids = range(1, 101)
    output_csv = recommend_all_to_csv(user_ids=user_ids, top_k=top_k)

    # S3 업로드
    bucket = os.getenv("S3_BUCKET_NAME")
    if bucket:
        kst = timezone(timedelta(hours=9))
        timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
        s3_key = f"inference_results/recommendations_{timestamp}.csv"
        print("📤 S3 업로드 시작")
        print("="*60)
        upload_to_s3(output_csv, bucket, s3_key)
    else:
        print("⚠️ S3_BUCKET_NAME 환경변수가 없습니다. 업로드 생략.")

    print("="*60)
    print("✅ 전체 작업 완료!")
    print("="*60)
    print(f"   로컬 파일: {output_csv}")
    if bucket:
        print(f"   S3 경로: s3://{bucket}/{s3_key}")
    print("="*60)

if __name__ == "__main__" and ("recommend" not in sys.argv and "recommend_all" not in sys.argv):
    main()
else:
    fire.Fire({
        "recommend": recommend,
        "recommend_all": recommend_all
    })

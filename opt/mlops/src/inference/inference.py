import os
import sys
import glob
import pickle
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

import pandas as pd
from tqdm import tqdm

sys.path.append( # /opt/mlops
    os.path.dirname(
		    os.path.dirname(
				    os.path.dirname(
						    os.path.abspath(__file__)
						)
				)
		)
)

from src.model.game_item_cf import ItemCF  # 같은 디렉토리 기준
from src.utils.utils import project_path, model_dir, download_latest_model_from_s3

# .env 로드
load_dotenv(os.path.join(project_path(), ".env"))

class ItemCFInference:
    def __init__(self, model_name: str, model_path: str = None, latest=True):
        self.model_name = model_name
        self.model_data = self.load_model(latest)
        self.model = ItemCF()
        # sim_matrix가 numpy array로 저장되어 있으므로 DataFrame으로 변환
        self.train_matrix = self.model_data["train_matrix"]
        self.model.item_similarity_df = pd.DataFrame(
            self.model_data["sim_matrix"],
            index=self.train_matrix.columns,
            columns=self.train_matrix.columns
        )
    
        print(f"✅ 모델 초기화 완료 - Train matrix shape: {self.train_matrix.shape}")

    def load_model(self, latest=True):
        """저장된 모델 중 최신 epoch 불러오기"""
        save_path = model_dir(self.model_name)
        files = [f for f in os.listdir(save_path) if f.endswith(".pkl")]
        if not files:
            raise FileNotFoundError("저장된 모델이 없습니다.")
        files.sort()  # 이름 기준 정렬
        target_file = files[-1] if latest else files[0]
        with open(os.path.join(save_path, target_file), "rb") as f:
            model_data = pickle.load(f)
        return model_data

    def recommend(self, user_id, top_k=5):
        """추천 결과 반환"""
        if user_id not in self.train_matrix.index:
            return []  # 없는 유저
        return self.model.recommend(user_id, self.train_matrix, top_k=top_k)

def recommend_all_to_csv(user_ids, top_k=5, popular_games_csv="/opt/data-prepare/result/popular_games.csv", output_dir="/opt/mlops/dataset/inference_results"):
    """
    1~100번 유저 추천 결과를 CSV로 생성
    """
    
    print("🎮 RECOMMEND ALL - 유저 {}-{}번 게임 추천".format(user_ids[0], user_ids[-1]))
    print("="*60)
    print("="*60)
    print("🚀 추론 시작")
    print("="*60)

    # 0. S3 환경 변수 확인
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise ValueError("S3_BUCKET_NAME 환경변수가 없습니다.")

    # 1. 최신 모델 다운로드
    local_dir = "/opt/mlops/models/itemCF"
    prefix = "models/itemCF"
    os.makedirs(local_dir, exist_ok=True)
    print(f"📩 S3에서 모델 로드 중: itemCF")
    local_model_path = download_latest_model_from_s3(
        bucket_name=bucket,
        prefix=prefix,
        local_dir=local_dir
    )

    # 2. Inference 객체 생성
    recommender = ItemCFInference(model_name="itemCF", model_path=local_model_path)

    # 3. popular_games.csv 로드
    print(f"📩 게임 정보 로드 중: {popular_games_csv}")
    popular_games = pd.read_csv(popular_games_csv)
    popular_games = popular_games[["game_id", "game_name", "rating", "genre"]]
    print(f"✅ 게임 정보 로드 완료: {len(popular_games)}개 게임")

    print(f"🔍 추론대상: User {user_ids[0]}~{user_ids[-1]} ({len(user_ids)}명)")
    print(f"   각 유저당 {top_k}개 게임 추천 (예상 총 {len(user_ids)*top_k}개 레코드)")

    # 4. 모든 유저 추천
    all_records = []
    success_users = 0
    for user_id in tqdm(user_ids, desc="추론 진행"):
        recommended_games = recommender.recommend(user_id, top_k=top_k)
        if recommended_games:
            success_users += 1
        for game_name in recommended_games:
            info = popular_games[popular_games["game_name"] == game_name]
            if not info.empty:
                row = info.iloc[0]
                all_records.append({
                    "user_id": user_id,
                    "game_id": row["game_id"],
                    "game_name": row["game_name"],
                    "rating": row["rating"],
                    "genre": row["genre"]
                })
            else:
                all_records.append({
                    "user_id": user_id,
                    "game_id": None,
                    "game_name": game_name,
                    "rating": None,
                    "genre": None
                })

    # 5. DataFrame 생성 및 CSV 저장
    df = pd.DataFrame(all_records)

    # 날짜·시간 기반 파일명 생성
    kst = timezone(timedelta(hours=9))
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"recommendations_{timestamp}.csv")

    df.to_csv(output_csv, index=False)
    print(f"✅ 추론 완료!")
    print(f"   생성된 레코드 수: {len(df)}개")
    print(f"   성공 유저 수: {success_users}/{len(user_ids)}명")
    print(f"   저장 경로: {output_csv}")
    print("="*60)
    print("📊 샘플 데이터 (처음 3개):")
    print(df.head(3))
    print("="*60)
    print(f"✅ Saved batch recommendations to {output_csv}")


    return output_csv

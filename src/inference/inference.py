# src/inference/inference.py

import os
import sys
import glob
import pickle
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from io import BytesIO
import boto3
import argparse
import logging


sys.path.append( # /opt/mlops
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)

from src.model.game_item_cf import ItemCF
from src.utils.utils import model_dir, load_model_from_s3


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# 기존 클래스 (로컬 모델만 사용)
# ============================================

class ItemCFInference:
    def __init__(self, model_name: str, latest=True):
        self.model_name = model_name
        
        # 로컬에서 모델 로드
        self.model_data = self.load_model(latest)
        self.model = ItemCF()
        
        # sim_matrix가 numpy array로 저장되어 있으므로 DataFrame으로 변환
        self.train_matrix = self.model_data["train_matrix"]
        self.model.item_similarity_df = pd.DataFrame(
            self.model_data["sim_matrix"],
            index=self.train_matrix.columns,
            columns=self.train_matrix.columns
        )

    def load_model(self, latest=True):
        """로컬에서 저장된 모델 로드"""
        save_path = model_dir(self.model_name)
        files = [f for f in os.listdir(save_path) if f.endswith(".pkl")]
        if not files:
            raise FileNotFoundError("저장된 모델이 없습니다.")
        files.sort()
        target_file = files[-1] if latest else files[0]
        
        with open(os.path.join(save_path, target_file), "rb") as f:
            model_data = pickle.load(f)
        return model_data

    def recommend(self, user_id, top_k=5):
        """단일 유저에 대한 추천 결과 반환 (game_name 리스트)"""
        if user_id not in self.train_matrix.index:
            return []
        
        recommendations = self.model.recommend(user_id, self.train_matrix, top_k=top_k)
        return recommendations


# ============================================
# 새로 추가: S3 기반 배치 추론 함수
# ============================================

def run_inference(bucket_name: str, model_name: str, game_info_path: str,
                 start_user: int, end_user: int, top_k: int, output_path: str):
    """
    S3에서 모델 로드하고 로컬 CSV로 추론 실행 (main.py recommend_all용)
    """
    
    print("=" * 60)
    print("🚀 추론 시작")
    print("=" * 60)
    
    # 1. S3에서 모델 로드 (무조건 S3)
    print(f"📥 S3에서 모델 로드 중: {model_name}")
    model_data = load_model_from_s3(bucket_name, model_name)
    
    # 2. 모델 초기화
    model = ItemCF()
    train_matrix = model_data["train_matrix"]
    model.item_similarity_df = pd.DataFrame(
        model_data["sim_matrix"],
        index=train_matrix.columns,
        columns=train_matrix.columns
    )
    print(f"✅ 모델 초기화 완료 - Train matrix shape: {train_matrix.shape}")
    
    # 3. 로컬에서 게임 정보 로드
    print(f"📥 로컬에서 게임 정보 로드 중: {game_info_path}")
    df_games = pd.read_csv(game_info_path)
    df_games.columns = df_games.columns.str.replace('﻿', '')
    df_games_unique = df_games.drop_duplicates(subset='game_name').copy()
    
    if 'rating' not in df_games_unique.columns:
        df_games_unique['rating'] = 8.0
    if 'genre' not in df_games_unique.columns:
        df_games_unique['genre'] = 'Action'
    
    game_dict = df_games_unique.set_index('game_name').to_dict('index')
    print(f"✅ 게임 정보 로드 완료: {len(game_dict)}개 게임")
    
    # 4. 배치 추론
    user_ids = list(range(start_user, end_user + 1))
    print(f"🔍 추론 대상: User {start_user}~{end_user} ({len(user_ids)}명)")
    print(f"   각 유저당 {top_k}개 게임 추천 (예상 총 {len(user_ids) * top_k}개 레코드)")
    
    results = []
    for user_id in tqdm(user_ids, desc="추론 진행"):
        if user_id not in train_matrix.index:
            continue
        
        recommendations = model.recommend(user_id, train_matrix, top_k=top_k)
        
        for game_name in recommendations:
            game_meta = game_dict.get(game_name)
            if game_meta is None:
                continue
            
            results.append({
                'user_id': user_id,
                'game_id': game_meta.get('game_id'),
                'game_name': game_name,
                'rating': game_meta.get('rating'),
                'genre': game_meta.get('genre')
            })
    
    # 5. 결과 저장
    df_results = pd.DataFrame(results)
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    df_results.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ 추론 완료!")
    print(f"   생성된 레코드 수: {len(df_results)}개")
    if len(df_results) > 0:
        print(f"   성공 유저 수: {df_results['user_id'].nunique()}/{len(user_ids)}명")
        print(f"   저장 경로: {output_path}")
        print(f"\n📊 샘플 데이터 (처음 3개):")
        print(df_results.head(5).to_string(index=False))
    
    return output_path


if __name__ == '__main__':
    # 기존 argparse는 그대로 유지 (필요시)
    pass

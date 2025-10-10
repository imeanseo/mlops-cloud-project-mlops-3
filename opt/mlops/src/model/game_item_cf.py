import os
import pickle
from datetime import datetime, timezone, timedelta
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.utils import model_dir

class ItemCF:
    """
    아이템 기반 협업필터링 모델
    """
    def __init__(self):
        self.item_similarity_df = None

    def fit(self, train_matrix: pd.DataFrame):
        """
        학습: 아이템 간 코사인 유사도 계산
        """
        similarity = cosine_similarity(train_matrix.T)
        self.item_similarity_df = pd.DataFrame(similarity, index=train_matrix.columns, columns=train_matrix.columns)
        return self

    def predict(self, user_vector):
        """
        유저 벡터를 받아 점수 계산
        """
        if self.item_similarity_df is None:
            raise ValueError("모델을 먼저 fit 하세요.")
        scores = user_vector @ self.item_similarity_df.values
        return pd.Series(scores, index=self.item_similarity_df.index)

    def recommend(self, user_id, train_matrix: pd.DataFrame, top_k=5):
        """
        특정 유저 top_k 추천
        """
        user_vector = train_matrix.loc[user_id].values
        scores = self.predict(user_vector)
        scores = scores[train_matrix.loc[user_id] == 0]  # 이미 본 게임 제외
        return scores.sort_values(ascending=False).head(top_k).index.tolist()

def model_save(model_name, sim_matrix, train_matrix, epoch, recall_history=None):
    """
    아이템 기반 CF 모델 저장
    Args:
        model_name: str, 모델 이름
        sim_matrix: np.array, item-item similarity
        train_matrix: np.array, train user-item matrix
        epoch: int, 현재 epoch
        recall_history: list, optional, epoch별 recall 기록
    """
    save_dir = model_dir(model_name)
    os.makedirs(save_dir, exist_ok=True)
    kst = timezone(timedelta(hours=9))
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(save_dir, f"{model_name}_E{epoch}_T{timestamp}.pkl")

    save_data = {
        "epoch": epoch,
        "sim_matrix": sim_matrix,
        "train_matrix": train_matrix,
        "recall_history": recall_history,
    }

    # 데이터 저장
    with open(dst, "wb") as f:
        pickle.dump(save_data, f)

    print(f"Model saved to {dst}")

    return dst

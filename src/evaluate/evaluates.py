import numpy as np
import pandas as pd

def compute_recall_at_k(train_matrix: pd.DataFrame, val_matrix: pd.DataFrame, item_similarity_df: pd.DataFrame, k=5):
    recalls = []
    for user_id in train_matrix.index:
        true_items = set(val_matrix.loc[user_id][val_matrix.loc[user_id] == 1].index)
        if len(true_items) == 0:
            continue
        user_vector = train_matrix.loc[user_id].values
        scores = user_vector @ item_similarity_df.values
        scores = pd.Series(scores, index=item_similarity_df.index)
        scores = scores[train_matrix.loc[user_id] == 0]  # 이미 본 게임 제외
        recommended = set(scores.sort_values(ascending=False).head(k).index)
        recalls.append(len(true_items & recommended) / len(true_items))
    return np.mean(recalls)

def recommend_items(user_id, train_matrix, item_similarity_df, top_k=5):
    user_vector = train_matrix.loc[user_id].values
    scores = user_vector @ item_similarity_df.values
    scores = pd.Series(scores, index=item_similarity_df.index)
    scores = scores[train_matrix.loc[user_id] == 0]
    return scores.sort_values(ascending=False).head(top_k).index.tolist()

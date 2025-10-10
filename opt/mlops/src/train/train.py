import wandb
import os
from src.utils.utils import init_seed, model_dir
from src.model.game_item_cf import ItemCF, model_save
from src.evaluate.evaluates import compute_recall_at_k
import pandas as pd

def train_model(train_matrix: pd.DataFrame, val_matrix: pd.DataFrame, n_epochs=10, project_name="game_recommendation"):
    # 시드 고정
    init_seed()
    
    recall_history = []
    model = ItemCF()
    
    for epoch in range(1, n_epochs+1):
        # 모델 학습
        model.fit(train_matrix)

        # 평가
        recall = compute_recall_at_k(train_matrix, val_matrix, model.item_similarity_df, k=5)
        recall_history.append(recall)
        wandb.log({"epoch": epoch, "Recall@5": recall})
        print(f"Epoch {epoch}: Recall@5 = {recall:.4f}")

    # 모델 저장
    model_path = model_save(
        model_name="itemCF",
        sim_matrix=model.item_similarity_df.values,
        train_matrix=train_matrix,
        epoch=n_epochs,
        recall_history=recall_history
    )

    return model, recall_history, model_path

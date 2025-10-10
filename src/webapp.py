import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.inference.inference import ItemCFInference

app = FastAPI()

# 정적 파일과 템플릿 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))  # /opt/mlops/src
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))
# app.mount("/static", StaticFiles(directory="mlops/src/static"), name="static")
# templates = Jinja2Templates(directory="mlops/src/templates")

# 게임 이름 ↔ ID 매핑
game_df = pd.read_csv("/opt/mlops/dataset/games_log.csv")
game_name_to_id = dict(zip(game_df["game_name"], game_df["game_id"]))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "games": []})

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, user_id: int = Form(...), top_k: int = Form(5)):
    recommender = ItemCFInference(model_name="itemCF")
    games = recommender.recommend(user_id=user_id, top_k=top_k)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "games": games,
            "user_id": user_id,
            "game_name_to_id": game_name_to_id
        }
    )

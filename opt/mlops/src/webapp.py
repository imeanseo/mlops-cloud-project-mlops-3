import os
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from fastapi import FastAPI, Request, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from dotenv import load_dotenv
import sqlalchemy as sa
from sqlalchemy.exc import OperationalError

# ─────────────────────────────────────────────────────────────
# 환경/경로 설정
# ─────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

load_dotenv(ROOT_DIR / ".env")

# ─────────────────────────────────────────────────────────────
# FastAPI 설정
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Game Recommender", version="1.0")

# 정적 파일 & 템플릿
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# ─────────────────────────────────────────────────────────────
# DB 연결
# ─────────────────────────────────────────────────────────────
MYSQL_HOST = os.getenv("MYSQL_HOST", "game-mlops-db")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "root")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "mlops")

DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)
engine = sa.create_engine(DATABASE_URL, pool_pre_ping=True)

# ─────────────────────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────────────────────
def get_recommendations(user_id: int) -> pd.DataFrame:
    """특정 사용자 추천 결과 조회"""
    try:
        with engine.connect() as conn:
            query = sa.text(
                """
                SELECT user_id, game_id, game_name, rating, genre
                FROM game_recommendations
                WHERE user_id = :user_id
                LIMIT 5;
                """
            )
            result = conn.execute(query, {"user_id": user_id}).fetchall()
    except OperationalError:
        raise HTTPException(status_code=500, detail="❌ MySQL 연결 실패. 설정을 확인하세요.")

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"❌ user_id={user_id} 에 대한 추천 결과가 없습니다. Airflow에서 MySQL 적재를 확인하세요."
        )

    df = pd.DataFrame(result, columns=result[0]._fields)
    # 이미지 경로 추가 (static/images/{game_id}.jpg)
    df["image_path"] = df["game_id"].apply(lambda gid: f"/static/images/{gid}.jpg")
    return df

# ─────────────────────────────────────────────────────────────
# 라우터
# ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """브라우저에서 index.html 렌더링"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, user_id: int = Form(...)):
    """HTML form용 추천 결과"""
    df = get_recommendations(user_id)
    recommendations: List[Dict[str, Any]] = df.to_dict(orient="records")
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "recommendations": recommendations, "user_id": user_id},
    )

@app.get("/api/recommendations")
async def api_recommendations(user_id: int = Query(...)):
    """JSON API (Swagger Docs에서 호출 가능)"""
    df = get_recommendations(user_id)
    return JSONResponse({"user_id": user_id, "items": df.to_dict(orient="records")})

@app.get("/healthz")
async def healthz():
    return {"ok": True}

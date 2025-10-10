from dotenv import load_dotenv
load_dotenv()

from crawler import fetch_games
from preprocessing import normalize_games, generate_synthetic_users

def main():
    results = fetch_games()          # 크롤러
    df = normalize_games(results)    # 전처리
    user_df = generate_synthetic_users(df, num_users=100, max_games=5, alpha=0.5, noise_scale=0.1, random_state=42)
    print(df.head())               # 정상 작동 여부 확인
    print(user_df.head())
    df.to_csv("./data-prepare/result/popular_games.csv", encoding="utf-8-sig", index=False)
    user_df.to_csv("./data-prepare/result/games_log.csv", encoding="utf-8-sig", index=False)

if __name__ == "__main__":
    main()

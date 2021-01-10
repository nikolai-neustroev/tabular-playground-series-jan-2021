import pandas as pd
import lightgbm as lgb
from loguru import logger

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_lgbm.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        params = read_params()

        df = pd.read_csv("data/interim/test_scaled.csv")
        model = lgb.Booster(model_file="models/lgbm/model.txt")

        features = params['features']

        df['target'] = model.predict(df[features])
        df[['id', 'target']].to_csv("data/submission/submission.csv", index=False)

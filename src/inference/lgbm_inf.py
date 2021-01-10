import pandas as pd
import lightgbm as lgb
from loguru import logger

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_lgbm.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        parameters = read_params()

        df = pd.read_csv("data/raw/test.csv")

        model_one = lgb.Booster(model_file="models/lgbm/model_one.txt")
        features_one = parameters['features_one']
        df['pred_one'] = model_one.predict(df[features_one])

        model_two = lgb.Booster(model_file="models/lgbm/model_two.txt")
        features_two = parameters['features_two']
        df['pred_two'] = model_two.predict(df[features_two])

        df['target'] = 0.5*df['pred_one'] + 0.5*df['pred_two']

        df[['id', 'target']].to_csv("data/submission/submission.csv", index=False)

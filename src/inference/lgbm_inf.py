import pandas as pd
import lightgbm as lgb
from loguru import logger

if __name__ == '__main__':
    logger.add("logs/logs_lgbm.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        df = pd.read_csv("data/raw/test.csv")
        model = lgb.Booster(model_file="models/lgbm/model.txt")

        features = [
            'cont1',
            'cont2',
            'cont3',
            'cont4',
            'cont5',
            'cont6',
            'cont7',
            'cont8',
            'cont9',
            'cont10',
            'cont11',
            'cont12',
            'cont13',
            'cont14',
        ]

        df['target'] = model.predict(df[features])
        df[['id', 'target']].to_csv("data/submission/submission.csv", index=False)

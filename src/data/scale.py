from loguru import logger
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_scale.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        params = read_params()
        train = pd.read_csv("data/raw/train.csv")

        features = params['features']

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(train[features])
        scaled = pd.DataFrame(data=scaled, columns=features)

        df_scaled = pd.concat([train[['id', 'target']], scaled], axis=1)
        df_scaled.to_csv("data/interim/train_scaled.csv", index=False)

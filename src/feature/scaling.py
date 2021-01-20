import pandas as pd
from loguru import logger
from sklearn.preprocessing import PowerTransformer

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_scaling.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        param = read_params()
        features = param['features']

        train = pd.read_csv("data/interim/train_logarithm.csv")
        test = pd.read_csv("data/interim/test_logarithm.csv")

        scaler = PowerTransformer()

        train[features] = scaler.fit_transform(train[features])
        test[features] = scaler.transform(test[features])

        train.to_csv("data/interim/train_scaled.csv", index=False)
        test.to_csv("data/interim/test_scaled.csv", index=False)

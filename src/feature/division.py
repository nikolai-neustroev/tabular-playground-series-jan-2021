import pandas as pd
from loguru import logger

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_scaling.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        param = read_params()
        features = param['features_to_div']

        train = pd.read_csv("data/interim/train_scaled.csv")
        test = pd.read_csv("data/interim/test_scaled.csv")

        for colname_x in features:
            others = [x for x in features if x != colname_x]
            for colname_y in others:
                new_col = colname_x + "_div_" + colname_y
                train[new_col] = train[colname_x] / train[colname_y]
                test[new_col] = test[colname_x] / test[colname_y]

        train.to_csv("data/interim/train_div.csv", index=False)
        test.to_csv("data/interim/test_div.csv", index=False)

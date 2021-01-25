import pandas as pd
from loguru import logger

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_add_features.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        param = read_params()
        additional_features = param['additional_features']

        train = pd.read_csv("data/interim/train_scaled.csv")
        test = pd.read_csv("data/interim/test_scaled.csv")

        # Add cont13_logarithm * cont4 feature
        train[additional_features[0]] = train['cont13_logarithm'] * train['cont4']
        test[additional_features[0]] = test['cont13_logarithm'] * test['cont4']

        train[additional_features[1]] = train['cont1'] - train['cont12']
        test[additional_features[1]] = test['cont1'] - test['cont12']

        train[additional_features[2]] = train['cont12'] / train['cont13_logarithm']
        test[additional_features[2]] = test['cont12'] / test['cont13_logarithm']

        train[additional_features[3]] = train['cont12'] + train['cont3']
        test[additional_features[3]] = test['cont12'] + test['cont3']


        train.to_csv("data/interim/train_add_features.csv", index=False)
        test.to_csv("data/interim/test_add_features.csv", index=False)

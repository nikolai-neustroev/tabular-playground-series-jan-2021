import pandas as pd
from loguru import logger
from sklearn.feature_selection import SelectKBest

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_feature_selection.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        param = read_params()
        features = param['features']
        k = param['k']

        train = pd.read_csv("data/interim/train_div.csv")
        print(f"DataFrame shape: {train.shape}")

        X = train[features]
        y = train['target']

        selector = SelectKBest(k=k)
        train_new = selector.fit_transform(X, y)

        cols = selector.get_support(indices=True)
        train_new = X.iloc[:, cols]

        print(f"DataFrame shape after: {train_new.shape}")

        train_new = pd.concat([train[['id', 'target']], train_new], axis=1)

        train_new.to_csv("data/interim/train_selected.csv", index=False)

        test = pd.read_csv("data/interim/test_div.csv")
        test_new = selector.transform(test[features])
        test_new = test[features].iloc[:, cols]
        test_new = pd.concat([test[['id']], test_new], axis=1)
        test_new.to_csv("data/interim/test_selected.csv", index=False)

import pandas as pd
from loguru import logger

if __name__ == '__main__':
    logger.add("logs/logs_drop_outliers.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        train = pd.read_csv("data/raw/train.csv")
        print(f"Original shape: {train.shape}")

        train = train[train['target'] >= 5.5]

        Q1 = train['cont7'].quantile(0.25)
        Q3 = train['cont7'].quantile(0.75)
        IQR = Q3 - Q1

        train = train[(train['cont7'] > (Q1 - 1.5 * IQR)) & (train['cont7'] < (Q3 + 1.5 * IQR))]
        train = train[(train['cont9'] > (Q1 - 1.5 * IQR)) & (train['cont9'] < (Q3 + 1.5 * IQR))]
        print(f"New shape: {train.shape}")

        train.to_csv("data/interim/train_no_outliers.csv")

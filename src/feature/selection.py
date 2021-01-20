from feature_selector import FeatureSelector
import pandas as pd
from loguru import logger

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_dfs.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        train_df = pd.read_csv('data/interim/train_dfs.csv')

        features = train_df.columns
        features = list(features)
        features.remove('id')
        features.remove('target')

        train = train_df[features]
        train_labels = train_df['target']

        fs = FeatureSelector(data=train, labels=train_labels)
        train_removed_all_once = fs.remove(methods='all')

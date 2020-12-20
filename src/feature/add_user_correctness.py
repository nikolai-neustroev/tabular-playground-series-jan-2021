import os
import sys

import pandas as pd
from loguru import logger

sys.path.append(os.getcwd())
from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_add_user_correctness.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        params = read_params()
        target = params['target']
        train_df = pd.read_csv("data/interim/train_prior_question_had_explanation_filled.csv")

        train_df['lag'] = train_df.groupby('user_id')[target].shift()
        cum = train_df.groupby('user_id')['lag'].agg(['cumsum', 'cumcount'])
        train_df['user_correctness'] = cum['cumsum'] / cum['cumcount']
        train_df.drop(columns=['lag'], inplace=True)

        print(train_df.head())
        train_df.to_csv("data/interim/train_with_user_correctness.csv")

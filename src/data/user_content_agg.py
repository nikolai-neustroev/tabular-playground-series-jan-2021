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

        user_agg = train_df.groupby('user_id')[target].agg(['sum', 'count'])
        print(user_agg.head())
        user_agg.to_csv("data/processed/user_agg.csv")

        content_agg = train_df.groupby('content_id')[target].agg(['sum', 'count'])
        print(content_agg.head())
        content_agg.to_csv("data/processed/content_agg.csv")

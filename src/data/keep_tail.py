import os
import sys

import pandas as pd
from loguru import logger

sys.path.append(os.getcwd())
from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_keep_tail.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        params = read_params()
        train_df = pd.read_csv("data/interim/train_prior_question_had_explanation_filled.csv")
        train_df = train_df.groupby('user_id').tail(params['tail_length']).reset_index(drop=True)
        print(train_df.head())
        train_df.to_csv("data/interim/train_tail.csv")

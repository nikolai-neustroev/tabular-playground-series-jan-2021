import os
import sys

import pandas as pd
from loguru import logger

sys.path.append(os.getcwd())
from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_train_val_split.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        params = read_params()
        train_df = pd.read_csv("data/interim/train_with_prior_question_elapsed_time_true.csv")

        valid_df = train_df.groupby('user_id').tail(6).copy()
        print(valid_df.head())
        print(valid_df.shape)

        train_df.drop(valid_df.index, inplace=True)
        print(train_df.head())

        train_df = train_df.reset_index()
        print(train_df.shape)
        print(train_df.info())

        train_df.to_csv("data/processed/train_df_mini.csv")
        valid_df.to_csv("data/processed/valid_df_mini.csv")

import os
import sys

import pandas as pd
from loguru import logger

sys.path.append(os.getcwd())
from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_add_lectures_watched_in_part.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        params = read_params()
        train_df = pd.read_csv("data/interim/train_cropped.csv")
        train_df = train_df.drop(columns=['task_container_id', 'user_answer'])
        target = params['target']
        train_df = train_df[train_df[target] != -1].reset_index(drop=True)  # drop lectures
        print(train_df.head())
        train_df.to_csv("data/interim/train_cropped_lectures_dropped.csv")

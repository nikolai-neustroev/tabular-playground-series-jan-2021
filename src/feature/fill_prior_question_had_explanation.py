import pandas as pd
from loguru import logger

if __name__ == '__main__':
    logger.add("logs/logs_fill_prior_question_had_explanation.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        train_df = pd.read_csv("data/interim/train_cropped_lectures_dropped.csv")
        train_df['prior_question_had_explanation'].fillna(0.0, inplace=True)
        print(train_df.head())
        train_df.to_csv("data/interim/train_prior_question_had_explanation_filled.csv")

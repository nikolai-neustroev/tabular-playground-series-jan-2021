import pandas as pd
from loguru import logger

if __name__ == '__main__':
    logger.add("logs/logs_add_prior_question_elapsed_time_true.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        train_df = pd.read_csv("data/interim/train_with_content_count.csv")

        train_df['prior_timestamp'] = train_df.groupby(['user_id'])['timestamp'].shift()
        train_df['prior_question_elapsed_time_true'] = train_df['timestamp'] - train_df['prior_timestamp']

        train_df['prior_question_elapsed_time_true'] = train_df['prior_question_elapsed_time_true'].fillna(0)
        train_df = train_df.drop(columns=['timestamp', 'prior_timestamp'])

        print(train_df.head())
        train_df.to_csv("data/interim/train_with_prior_question_elapsed_time_true.csv")

import pandas as pd
from loguru import logger

if __name__ == '__main__':
    logger.add("logs/logs_add_bundle_id_and_part.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        questions_df = pd.read_csv("data/raw/questions.csv")
        train_df = pd.read_csv("data/interim/train_tail.csv")
        train_df = pd.merge(train_df, questions_df[['question_id', 'bundle_id', 'part']], left_on='content_id',
                            right_on='question_id', how='left')
        train_df.drop(columns=['question_id'], inplace=True)
        print(train_df.head())
        train_df.to_csv("data/interim/train_with_bundle_id_part.csv")

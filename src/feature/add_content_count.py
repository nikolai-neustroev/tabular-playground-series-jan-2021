import pandas as pd
from loguru import logger

if __name__ == '__main__':
    logger.add("logs/logs_add_content_count.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        train_df = pd.read_csv("data/interim/train_with_lecture_count.csv")
        content_agg = pd.read_csv("data/processed/content_agg.csv")

        train_df['content_count'] = train_df['content_id'].map(content_agg['count']).astype('int32')
        train_df['content_id'] = train_df['content_id'].map(content_agg['sum'] / content_agg['count'])

        print(train_df.head())
        train_df.to_csv("data/interim/train_with_content_count.csv")

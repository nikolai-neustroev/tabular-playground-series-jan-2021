import pandas as pd
from loguru import logger


def crop_train_set(train_df: pd.DataFrame, frac: float, sample_seed: int) -> pd.DataFrame:
    user_pool = train_df['user_id'].unique()
    user_pool = pd.Series(user_pool)
    user_pool = user_pool.sample(frac=frac, random_state=sample_seed)
    train_df = train_df[train_df['user_id'].isin(user_pool)]
    return train_df


if __name__ == '__main__':
    logger.add("logs/logs_sample_data.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        df = pd.read_feather("data/raw/train.feather")
        print(f"Train set shape before: {df.shape}")
        df = crop_train_set(df, 1/80, 3)
        print(f"Train set shape after: {df.shape}")

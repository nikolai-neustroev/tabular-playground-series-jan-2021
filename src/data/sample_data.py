import os
import sys

import pandas as pd
from loguru import logger

sys.path.append(os.getcwd())
from src.utils.read_params import read_params

def crop_train_set(train_df: pd.DataFrame, frac: float, sample_seed: int) -> pd.DataFrame:
    """Returns cropped train set containing a certain fraction of users only.

    Parameters
    ----------
    train_df : pd.DataFrame
        A Pandas DataFrame to crop.
    frac : float
        Fraction of axis items to return.
    sample_seed : int
        Seed for random number generator.

    Returns
    -------
     train_df : pd.DataFrame
        Cropped train set.

    """
    user_pool = train_df['user_id'].unique()
    user_pool = pd.Series(user_pool)
    user_pool = user_pool.sample(frac=frac, random_state=sample_seed)
    train_df = train_df[train_df['user_id'].isin(user_pool)]
    return train_df


if __name__ == '__main__':
    logger.add("logs/logs_sample_data.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        params = read_params()
        df = pd.read_feather("data/raw/train.feather")
        print(f"Train set shape before: {df.shape}")
        df = crop_train_set(df, params['frac'], params['seed'])
        print(f"Train set shape after: {df.shape}")
        df.to_csv("data/interim/train_cropped.csv")

import pandas as pd
from loguru import logger


def read_train_csv(path: str) -> pd.DataFrame:
    """Read train.csv

    path : str
        Path to train.csv file.

    Returns
    -------
    train_df : pd.DataFrame
        train.csv as Pandas DataFrame
    """
    data_types_dict = {
        'row_id': 'int64',
        'timestamp': 'int64',
        'user_id': 'int32',
        'content_id': 'int16',
        'content_type_id': 'int8',
        'task_container_id': 'int16',
        'user_answer': 'int8',
        'answered_correctly': 'int8',
        'prior_question_elapsed_time': 'float32',
        'prior_question_had_explanation': 'int8'

    }

    train_df = pd.read_csv(
        path,
        dtype=data_types_dict
    )

    return train_df


if __name__ == '__main__':
    logger.add("logs/logs_read_train_csv.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        df = read_train_csv('data/raw/train.csv')
        print(df.info())

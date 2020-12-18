import os
import sys

from loguru import logger

sys.path.append(os.getcwd())
from src.utils.read_train_csv import read_train_csv

if __name__ == '__main__':
    logger.add("logs/logs_csv2feather.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        sys.path.append(os.getcwd())
        train_csv = read_train_csv('data/raw/train.csv')
        train_csv.to_feather('data/raw/train.feather')

import pandas as pd
from gmr import GMM
from loguru import logger
import numpy as np

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_gmr.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        params = read_params()
        features = params['features']

        # Train
        train = pd.read_csv("data/interim/train_cleaned.csv")
        gmm = GMM(n_components=2, random_state=42)
        print(train[features])
        gmm.from_samples(train[features])

        # Inference
        test = pd.read_csv("data/interim/test_cleaned.csv")
        test['target'] = gmm.predict(np.array([0]), test[features])
        test[['id', 'target']].to_csv("data/submission/submission.csv", index=False)

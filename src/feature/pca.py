from loguru import logger
import pandas as pd
from sklearn.decomposition import PCA

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_pca.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        params = read_params()
        train = pd.read_csv("data/interim/train_augmented.csv")

        features = train.columns
        features = list(features)
        features.remove('id')
        features.remove('target')

        pca = PCA(n_components=params['n_components']).fit(train[features])
        df_pca = pd.DataFrame(pca.transform(train[features]))
        df_pca = df_pca.add_prefix('pca_')

        df_pca = pd.concat([train[['id', 'target']], df_pca], axis=1)
        df_pca.to_csv("data/interim/train_pca.csv", index=False)

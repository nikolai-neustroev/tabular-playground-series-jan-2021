import pandas as pd
import featuretools as ft
from feature_selector import FeatureSelector
from loguru import logger

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_dfs.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        param = read_params()
        features = param['features']
        # Can't keep all of them since memory is restricted.
        COLS_TO_TRANSFORM = features  # [0:3]
        # Here, I limit the depth since memory is restricted.
        MAX_DEPTH = param['max_depth']

        train_df = pd.read_csv('data/interim/train_scaled.csv')
        # train_df = train_df.head(100)

        es = ft.EntitySet()
        es = es.entity_from_dataframe(
            entity_id="features",
            dataframe=train_df[COLS_TO_TRANSFORM],
            index="index",
        )

        # To see the available transform primitives: ft.primitives.list_primitives()
        # Trying some random ones here...

        TRANSFORM_PRIMITIVES = param['transform_primitives']

        # Use Dask for execution and speedup.
        augmented_train_df, _ = ft.dfs(entityset=es, target_entity="features",
                                       trans_primitives=TRANSFORM_PRIMITIVES, max_depth=MAX_DEPTH)

        print(augmented_train_df.shape)

        # augmented_train_df = pd.concat([train_df[['id', 'target']], augmented_train_df], axis=1)

        # augmented_train_df.to_csv("data/interim/train_dfs.csv", index=False)

        # train_df = pd.read_csv('data/interim/train_dfs.csv')

        # features = train_df.columns
        # features = list(features)
        # features.remove('id')
        # features.remove('target')
        #
        # train = train_df[features]
        # train_labels = train_df['target']

        fs = FeatureSelector(data=augmented_train_df, labels=train_df['target'])
        fs.identify_all(selection_params={
            'missing_threshold': 0.6,
            'correlation_threshold': 0.98,
            'task': 'regression',
            'eval_metric': 'rmse',
            'early_stopping': False,
            'cumulative_importance': 0.95,
        })
        train_removed_all_once = fs.remove(methods='all')
        train_removed_all_once = pd.concat([train_df[['id', 'target']], train_removed_all_once], axis=1)
        train_removed_all_once.to_csv("data/interim/train_augmented.csv", index=False)

        test_df = pd.read_csv('data/interim/test_scaled.csv')
        # test_df = test_df.head(100)
        es = ft.EntitySet()
        es = es.entity_from_dataframe(
            entity_id="features",
            dataframe=test_df[COLS_TO_TRANSFORM],
            index="index",
        )
        augmented_test_df, _ = ft.dfs(entityset=es, target_entity="features",
                                      trans_primitives=TRANSFORM_PRIMITIVES, max_depth=MAX_DEPTH)
        augmented_test_df = augmented_test_df.drop(columns=fs.removed_features)
        augmented_test_df = pd.concat([test_df[['id']], augmented_test_df], axis=1)
        augmented_test_df.to_csv("data/interim/test_augmented.csv", index=False)

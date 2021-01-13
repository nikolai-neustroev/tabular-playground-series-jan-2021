import json

import pandas as pd
import lightgbm as lgb
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import neptune
from neptunecontrib.monitoring.lightgbm import neptune_monitor

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_lgbm.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        with open("params/neptune.json") as json_file:
            neptune_init = json.load(json_file)
        neptune.init(
            project_qualified_name=neptune_init['project_name'],
            api_token=neptune_init['api_token'],
        )

        param = read_params()

        df = pd.read_csv("data/interim/train_scaled.csv")
        train_df, valid_df = train_test_split(df, test_size=0.4, random_state=42)

        target = 'target'

        features = param['features']

        params = {
            'objective': 'regression',
            'seed': 42,
            'metric': 'rmse',
            'learning_rate': 0.05,
            'max_bin': 800,
            'num_leaves': 80,
            'min_child_samples': train_df.shape[0] // 50,
        }

        tr_data = lgb.Dataset(train_df[features], label=train_df[target])
        va_data = lgb.Dataset(valid_df[features], label=valid_df[target])

        evals_result = {}  # Record training results
        experiment = neptune.create_experiment(
            name='lgb',
            tags=['train'],
            params=params,
            properties={'target': target, 'features': ', '.join(features)}
        )
        monitor = neptune_monitor()

        model = lgb.train(
            params,
            tr_data,
            num_boost_round=5000,
            valid_sets=[tr_data, va_data],
            evals_result=evals_result,
            early_stopping_rounds=100,
            verbose_eval=100,
            callbacks=[monitor],
        )

        model.save_model('models/lgbm/model.txt')
        ax = lgb.plot_importance(model, importance_type='gain')
        experiment.log_image('importance', plt.gcf())

        neptune.stop()

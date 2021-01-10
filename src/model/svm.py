# import json

# import neptune
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import LinearSVR

if __name__ == '__main__':
    logger.add("logs/logs_lgbm.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        # with open("params/neptune.json") as json_file:
        #     neptune_init = json.load(json_file)
        # neptune.init(
        #     project_qualified_name=neptune_init['project_name'],
        #     api_token=neptune_init['api_token'],
        # )

        train_df = pd.read_csv("data/raw/train.csv")

        target = 'target'

        features = [
            'cont1',
            'cont2',
            'cont3',
            'cont4',
            'cont5',
            'cont6',
            'cont7',
            'cont8',
            'cont9',
            'cont10',
            'cont11',
            'cont12',
            'cont13',
            'cont14',
        ]

        svr = LinearSVR(
            epsilon=0.6,
            random_state=42,
        )

        print("Cross validation started")

        scores = cross_val_score(
            svr,
            train_df[features],
            train_df[target],
            cv=3,
            scoring='neg_root_mean_squared_error',
        )

        print(scores)

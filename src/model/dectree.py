import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from loguru import logger

if __name__ == '__main__':
    logger.add("logs/logs_dectree.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        df = pd.read_csv("data/raw/train.csv")

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

        min_samples_leaf = df.shape[0] // 50

        clf = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf, random_state=42)
        clf.fit(df[features], df[target])

        test = pd.read_csv("data/raw/test.csv")
        test['target'] = clf.predict(test[features])

        test[['id', 'target']].to_csv("data/submission/submission.csv", index=False)

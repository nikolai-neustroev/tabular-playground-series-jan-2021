import argparse
from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_scale.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        params = read_params()

        parser = argparse.ArgumentParser()
        parser.add_argument("--file", "-f", help="CSV to scale")
        args = parser.parse_args()
        p = Path(args.file)

        df = pd.read_csv(p)

        features = params['features']

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[features])
        scaled = pd.DataFrame(data=scaled, columns=features)

        unwanted = df.columns[df.columns.str.startswith('cont')]
        df_not_features = df.drop(unwanted, axis=1)
        # df_not_features = df[df.columns.difference(features)]

        df_scaled = pd.concat([df_not_features, scaled], axis=1)
        df_scaled.to_csv(f"data/interim/{p.stem}_scaled.csv", index=False)

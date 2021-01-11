import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

if __name__ == '__main__':
    logger.add("logs/logs_drop_outliers.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        parser = argparse.ArgumentParser()
        parser.add_argument("--file", "-f", help="CSV to scale")
        args = parser.parse_args()
        p = Path(args.file)

        df = pd.read_csv(p)

        Q1 = df['target'].quantile(0.25)
        Q3 = df['target'].quantile(0.75)
        IQR = Q3 - Q1  # IQR is interquartile range.

        filter = (df['target'] >= Q1 - 1.5 * IQR) & (df['target'] <= Q3 + 1.5 * IQR)
        df = df.loc[filter]

        df.to_csv(f"data/interim/{p.stem}_cleaned.csv", index=False)

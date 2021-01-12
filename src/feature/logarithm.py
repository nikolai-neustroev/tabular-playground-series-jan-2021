import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.read_params import read_params

if __name__ == '__main__':
    logger.add("logs/logs_logarithm.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        param = read_params()

        parser = argparse.ArgumentParser()
        parser.add_argument("--file", "-f", help="CSV to scale")
        args = parser.parse_args()
        p = Path(args.file)

        df = pd.read_csv(p)

        for colname in param['features_to_logarithm']:
            df[f'{colname}_logarithm'] = np.log(df[colname])

        df.to_csv(f"data/interim/{p.stem}_logarithm.csv", index=False)

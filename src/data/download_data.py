import os
import sys
import zipfile
from typing import NoReturn

import kaggle
from loguru import logger

sys.path.append(os.getcwd())

from src.utils.read_params import read_params


def get_data(dataset: str, path: str) -> NoReturn:
    """Authenticate and download Kaggle dataset.

    Parameters
    ----------
    dataset : str
        Kaggle dataset name in "username/dataset_name" format.
    path : str
        Directory path.

    """
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(dataset, path=path)


def unzip(outfile: str, effective_path: str) -> NoReturn:
    """Extract files from a ZIP archive and removes the ZIP file.

    Parameters
    ----------
    outfile : str
        ZIP file path.
    effective_path : str
        Extraction target directory.

    """
    try:
        with zipfile.ZipFile(outfile) as z:
            z.extractall(effective_path)
    except zipfile.BadZipFile as e:
        raise ValueError(
            'Bad zip file, please report on '
            'www.github.com/kaggle/kaggle-api', e)
    try:
        os.remove(outfile)
    except OSError as e:
        print('Could not delete zip file, got %s' % e)


if __name__ == '__main__':
    logger.add("logs/logs_download_data.txt", level="TRACE", rotation="10 KB")
    with logger.catch():
        params = read_params()
        get_data(params['kaggle_competition'], 'data/raw')
        unzip(f"data/raw/{params['kaggle_competition']}.zip", "data/raw")

"""Download the Cleveland Heart Disease data from UCI and create the cleaned CSV.

Usage:  python -m src.download
Manual alternative: download `processed.cleveland.data` from
https://archive.ics.uci.edu/dataset/45/heart+disease and place it in data/raw/.
"""
import urllib.request

import pandas as pd

from src.config import CLEAN_PATH, RAW_COLUMNS, RAW_PATH
from src.data import clean, load_raw

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)


def download(dest=RAW_PATH):
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(URL, dest)
        if len(load_raw(dest)) < 250:
            raise ValueError("downloaded file looks incomplete")
        print(f"Downloaded raw data to {dest}")
    except Exception as exc:  # fall back to the ucimlrepo package
        print(f"Direct download failed ({exc}); trying ucimlrepo ...")
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=45)
        df = pd.concat([data.data.features, data.data.targets], axis=1)
        df[RAW_COLUMNS].to_csv(dest, header=False, index=False)
        print(f"Saved raw data via ucimlrepo to {dest}")


def main():
    download()
    df = clean(load_raw())
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned dataset: {CLEAN_PATH} shape={df.shape}")
    print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()

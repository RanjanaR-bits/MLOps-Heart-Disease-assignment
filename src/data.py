"""Loading and cleaning the Cleveland heart-disease data."""
import pandas as pd

from src.config import CLEAN_PATH, RAW_COLUMNS, RAW_PATH, TARGET


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    """Read the raw UCI file ('?' marks missing values)."""
    return pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to numeric, drop duplicates and turn `num` (0-4) into a binary target.

    Missing values are intentionally kept here and imputed inside the model
    pipeline, so imputation statistics are learned from training folds only.
    """
    out = df.apply(pd.to_numeric, errors="coerce").drop_duplicates()
    out[TARGET] = (out["num"] > 0).astype(int)
    return out.drop(columns="num").reset_index(drop=True)


def load_clean(path=CLEAN_PATH):
    """Return (X, y) from the cleaned CSV."""
    df = pd.read_csv(path)
    return df.drop(columns=TARGET), df[TARGET]


def main():
    df = clean(load_raw())
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved {CLEAN_PATH} with shape {df.shape}")


if __name__ == "__main__":
    main()

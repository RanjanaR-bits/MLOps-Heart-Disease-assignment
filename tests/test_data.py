import numpy as np
import pandas as pd

from src.config import RAW_COLUMNS, TARGET
from src.data import clean


def _raw_rows():
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, "3", 3, 2],
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, "?", 4],
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_target_is_binary():
    out = clean(_raw_rows())
    assert set(out[TARGET].unique()) <= {0, 1}
    assert out[TARGET].tolist() == [0, 1, 1]
    assert "num" not in out.columns


def test_question_marks_become_nan():
    out = clean(_raw_rows())
    assert np.isnan(out.loc[2, "thal"])
    assert out.loc[1, "ca"] == 3  # the string "3" is coerced to a number


def test_duplicates_removed():
    raw = pd.concat([_raw_rows(), _raw_rows().iloc[[0]]], ignore_index=True)
    assert len(clean(raw)) == 3

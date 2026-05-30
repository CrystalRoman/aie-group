from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < val_size < 1 or not 0 < test_size < 1:
        raise ValueError("val_size and test_size must be in (0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")

    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=df["has_pneumo"],
    )

    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_df["has_pneumo"],
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

from pathlib import Path

import pandas as pd

INDEX_COLUMN = "Unnamed: 0"


def load_german_credit_data(path: str | Path) -> pd.DataFrame:
    """Load the German Credit dataset and drop the spurious index column."""
    df = pd.read_csv(path)
    if INDEX_COLUMN in df.columns:
        df = df.drop(columns=[INDEX_COLUMN])
    return df
import pandas as pd

NO_ACCOUNT_VALUE = "None"
RISK_MAPPING = {"good": 0, "bad": 1}


def impute_account_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in account columns, meaning the client has no account."""
    df = df.copy()
    df["Saving accounts"] = df["Saving accounts"].fillna(NO_ACCOUNT_VALUE)
    df["Checking account"] = df["Checking account"].fillna(NO_ACCOUNT_VALUE)
    return df


def encode_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Map the target Risk column to numeric (good=0, bad=1)."""
    df = df.copy()
    df["Risk_num"] = df["Risk"].map(RISK_MAPPING)
    return df
"""Step 1 - clean the raw CSV so the rest of the pipeline can trust it."""

import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from modules.utils import get_id_columns

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "?"}


def _is_text(series):
    return not is_numeric_dtype(series) and not is_bool_dtype(series) and (
        pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    )


def clean_data(df):
    """Return a cleaned copy of ``df``.

    What it does: tidy column names, trim text, turn "N/A"-style words into real
    missing values, convert numbers stored as text, drop empty rows/columns and
    duplicate rows, then fill missing values (median for numbers, most common
    value for text).

    A log of every change is kept in ``df.attrs["cleaning_log"]`` and the missing
    values found *before* filling in ``df.attrs["missing_before"]``.
    """
    log = []
    df = df.copy()

    # 1. Column names: "Study Hours " -> "Study_Hours"
    df.columns = [re.sub(r"\s+", "_", str(c).strip()) for c in df.columns]

    # 2. Empty rows / columns
    rows_before, cols_before = df.shape
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.shape[0] < rows_before:
        log.append(f"Removed {rows_before - df.shape[0]} completely empty row(s).")
    if df.shape[1] < cols_before:
        log.append(f"Removed {cols_before - df.shape[1]} completely empty column(s).")

    # 3. Text columns: trim spaces, treat "N/A", "null" ... as missing
    for col in df.columns:
        if _is_text(df[col]):
            cleaned = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            cleaned = cleaned.map(
                lambda v: np.nan if isinstance(v, str) and v.lower() in MISSING_TOKENS else v
            )
            df[col] = cleaned

    # 4. Numbers stored as text ("85", "12.5") -> numeric
    for col in df.columns:
        if _is_text(df[col]):
            non_null = df[col].notna().sum()
            if non_null == 0:
                continue
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= 0.9 * non_null:
                lost = int(non_null - converted.notna().sum())
                df[col] = converted
                msg = f"Converted column '{col}' from text to numbers."
                if lost:
                    msg += f" {lost} non-numeric value(s) became missing."
                log.append(msg)

    # 5. Duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        log.append(f"Removed {before - len(df)} duplicate row(s).")

    df = df.reset_index(drop=True)

    # 6. Missing values (remember what was missing, then fill)
    missing_before = {c: int(n) for c, n in df.isna().sum().items()}
    id_columns = set(get_id_columns(df))
    for col in df.columns:
        n_missing = missing_before[col]
        if n_missing == 0 or col in id_columns:
            continue
        if is_numeric_dtype(df[col]) and not is_bool_dtype(df[col]):
            fill = df[col].median()
            how = "median"
        else:
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            how = "most common value"
        df[col] = df[col].fillna(fill)
        log.append(f"Filled {n_missing} missing value(s) in '{col}' with the {how} ({fill}).")

    df.attrs["cleaning_log"] = log
    df.attrs["missing_before"] = missing_before
    return df

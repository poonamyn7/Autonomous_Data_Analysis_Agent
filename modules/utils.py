"""Shared helpers used by every module of the agent."""

import re

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

# Column names that look like identifiers: Student_ID, id, ID_number, StudentID, index ...
_ID_PATTERN = re.compile(r"(^|[_\s\-])id$|^id([_\s\-]|$)|^index$|^unnamed", re.IGNORECASE)
_CAMEL_ID_PATTERN = re.compile(r"[a-z]ID$")
_NOT_ID_WORDS = ("year", "month", "day", "age", "week")


def get_id_columns(df):
    """Return columns that identify a row (they must not be analysed as data)."""
    ids = []
    for col in df.columns:
        name = str(col)
        if _ID_PATTERN.search(name) or _CAMEL_ID_PATTERN.search(name):
            ids.append(col)
            continue
        series = df[col]
        if (
            is_integer_dtype(series)
            and len(series) >= 5
            and series.is_unique
            and not any(w in name.lower() for w in _NOT_ID_WORDS)
        ):
            steps = series.sort_values().diff().dropna()
            if (steps == 1).all():  # 1, 2, 3, 4 ... running number
                ids.append(col)
    return ids


def get_numeric_columns(df, exclude_ids=True, exclude_constant=True):
    """Numeric columns that are worth analysing."""
    ids = set(get_id_columns(df)) if exclude_ids else set()
    columns = []
    for col in df.columns:
        series = df[col]
        if col in ids or is_bool_dtype(series) or not is_numeric_dtype(series):
            continue
        if exclude_constant and series.nunique(dropna=True) <= 1:
            continue
        columns.append(col)
    return columns


def get_categorical_columns(df, max_unique=30):
    """Text / boolean columns with a small number of repeated values."""
    ids = set(get_id_columns(df))
    columns = []
    for col in df.columns:
        series = df[col]
        if col in ids or is_datetime64_any_dtype(series):
            continue
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            continue
        unique = series.nunique(dropna=True)
        if 1 < unique <= max_unique and unique < len(df):
            columns.append(col)
    return columns


def describe_correlation(r):
    """Turn a correlation coefficient into plain English, e.g. 'strong positive'."""
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "undefined"
    size = abs(r)
    if size >= 0.7:
        strength = "strong"
    elif size >= 0.4:
        strength = "moderate"
    elif size >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"
    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"


def strongest_correlation(corr):
    """Return (col_a, col_b, r) for the largest absolute off-diagonal correlation."""
    if corr is None or corr.shape[0] < 2:
        return None
    best = None
    columns = list(corr.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            r = corr.loc[a, b]
            if pd.isna(r):
                continue
            if best is None or abs(r) > abs(best[2]):
                best = (a, b, float(r))
    return best


def row_label(record, id_columns):
    """A readable name for a record, e.g. 'Student_ID 12' or 'row 7'."""
    for col in id_columns:
        if col in record:
            return f"{col} {record[col]}"
    if "_row" in record:
        return f"row {record['_row']}"
    return "a record"


def explain_anomaly_record(record, df, numeric_columns):
    """Name the column in which an anomalous record deviates most from the rest."""
    best_col, best_z = None, 0.0
    for col in numeric_columns:
        std = df[col].std()
        if not std or pd.isna(std) or col not in record:
            continue
        z = (record[col] - df[col].mean()) / std
        if abs(z) > abs(best_z):
            best_col, best_z = col, z
    if best_col is None:
        return "unusual combination of values"
    direction = "above" if best_z > 0 else "below"
    return (
        f"{best_col} = {record[best_col]:g} is {abs(best_z):.1f} standard "
        f"deviations {direction} the average"
    )

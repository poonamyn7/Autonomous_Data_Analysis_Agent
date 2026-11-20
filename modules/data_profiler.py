"""Step 2 - describe the dataset's structure."""

from modules.utils import (
    get_categorical_columns,
    get_id_columns,
    get_numeric_columns,
)


def profile_data(df):
    """Return a dictionary describing the (cleaned) dataset."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values": {c: int(n) for c, n in df.isna().sum().items()},
        "data_types": {c: str(t) for c, t in df.dtypes.items()},
        "id_columns": get_id_columns(df),
        "numeric_columns": get_numeric_columns(df),
        "categorical_columns": get_categorical_columns(df),
        "duplicate_rows": int(df.duplicated().sum()),
        # filled in by the cleaner - what was missing *before* it was repaired
        "missing_before_cleaning": dict(df.attrs.get("missing_before", {})),
        "cleaning_log": list(df.attrs.get("cleaning_log", [])),
    }

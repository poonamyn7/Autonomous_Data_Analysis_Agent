"""Step 3 - the agent's brain: look at the dataset and decide what to run.

The agent is rule-based and fully explainable: every decision comes with a
plain-English reason that is shown in the app.
"""

from modules.utils import (
    get_categorical_columns,
    get_id_columns,
    get_numeric_columns,
)

MIN_ROWS_FOR_ML = 30      # below this a model would just memorise the data
MIN_ROWS_FOR_ANOMALY = 8  # below this "unusual" has no meaning

# Words that usually mark the column a person wants to predict (checked in order)
TARGET_KEYWORDS = [
    "marks", "score", "grade", "result", "price", "salary", "income",
    "revenue", "sales", "profit", "cost", "amount", "rating", "target",
    "label", "total",
]


def pick_target(df, numeric_columns):
    """Choose the numeric column most likely to be the prediction target."""
    candidates = [c for c in numeric_columns if df[c].nunique() >= 5]
    for keyword in TARGET_KEYWORDS:
        for col in candidates:
            if keyword in col.lower():
                return col, f"its name contains '{keyword}'"
    if candidates:
        return candidates[-1], "it is the last numeric column with enough distinct values"
    return None, None


def analyze_dataset(df, target_override=None):
    """Return the plan: which analyses to run and why.

    ``target_override`` lets the person choose the target column instead of
    letting the agent guess it.
    """
    reasons = []
    rows = len(df)
    id_columns = get_id_columns(df)
    numeric = get_numeric_columns(df)
    categorical = get_categorical_columns(df)

    if id_columns:
        reasons.append(
            f"Identifier column(s) {', '.join(map(str, id_columns))} were detected and are excluded "
            "from statistics, machine learning and anomaly detection."
        )

    # ---- EDA ------------------------------------------------------------
    run_eda = rows >= 2 and (len(numeric) + len(categorical)) >= 1
    if run_eda:
        reasons.append(
            f"EDA will run: the dataset has {len(numeric)} numeric and "
            f"{len(categorical)} categorical column(s) to summarise."
        )
    else:
        reasons.append("EDA is skipped: there is not enough usable data to summarise.")

    # ---- Machine learning -----------------------------------------------
    if target_override in numeric:
        target, target_reason = target_override, "you selected it as the target"
    else:
        target, target_reason = pick_target(df, numeric)
    n_features = len([c for c in df.columns if c != target and c not in id_columns])
    ml_skip_reason = None

    if target is None:
        ml_skip_reason = "no numeric column with enough distinct values to use as a target."
    elif rows < MIN_ROWS_FOR_ML:
        ml_skip_reason = (
            f"the dataset has only {rows} rows (at least {MIN_ROWS_FOR_ML} are needed "
            "for a reliable train/test split)."
        )
    elif n_features < 1:
        ml_skip_reason = "there are no feature columns besides the target."

    run_ml = ml_skip_reason is None
    if run_ml:
        reasons.append(
            f"Machine Learning will run: '{target}' was selected as the target because "
            f"{target_reason}, and {rows} rows is enough to train and test a model."
        )
    else:
        reasons.append(f"Machine Learning is skipped: {ml_skip_reason}")
        target = target if target else None

    # ---- Anomaly detection ----------------------------------------------
    run_anomaly = len(numeric) >= 1 and rows >= MIN_ROWS_FOR_ANOMALY
    if run_anomaly:
        reasons.append(
            f"Anomaly Detection will run on {len(numeric)} numeric column(s) "
            f"across {rows} rows."
        )
    else:
        reasons.append(
            "Anomaly Detection is skipped: it needs at least "
            f"{MIN_ROWS_FOR_ANOMALY} rows and one numeric column."
        )

    return {
        "run_eda": run_eda,
        "run_ml": run_ml,
        "run_anomaly_detection": run_anomaly,
        "target_column": target,
        "reasons": reasons,
        "ml_skip_reason": ml_skip_reason,
        "id_columns": id_columns,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
    }

"""Step 4 - carry out the plan made by the agent."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from modules.utils import strongest_correlation

BLUE = "#2F6FED"
ORANGE = "#F28C28"
MAX_HISTOGRAMS = 6
MAX_BAR_CHARTS = 3


# ----------------------------------------------------------------------
# Charts (matplotlib Figure objects, no pyplot -> no memory leaks)
# ----------------------------------------------------------------------
def _histogram(df, col):
    fig = Figure(figsize=(5, 3.2))
    ax = fig.subplots()
    bins = int(min(15, max(5, np.sqrt(len(df)))))
    ax.hist(df[col].dropna(), bins=bins, color=BLUE, edgecolor="white")
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def _bar_chart(df, col):
    counts = df[col].value_counts().head(10)
    fig = Figure(figsize=(5, 3.2))
    ax = fig.subplots()
    ax.bar(counts.index.astype(str), counts.values, color=ORANGE)
    ax.set_title(f"Count by {col}")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def _heatmap(corr):
    size = max(4.5, 0.8 * len(corr))
    fig = Figure(figsize=(size, size * 0.85))
    ax = fig.subplots()
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig


def _scatter(df, a, b):
    fig = Figure(figsize=(5, 3.2))
    ax = fig.subplots()
    ax.scatter(df[a], df[b], color=BLUE, alpha=0.7)
    if df[a].nunique() > 1:
        slope, intercept = np.polyfit(df[a], df[b], 1)
        xs = np.linspace(df[a].min(), df[a].max(), 50)
        ax.plot(xs, slope * xs + intercept, color=ORANGE, linewidth=2)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f"{b} vs {a}")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------
def run_eda(df, numeric, categorical):
    eda = {
        "summary_statistics": df[numeric].describe() if numeric else pd.DataFrame(),
        "correlation": df[numeric].corr() if len(numeric) >= 2 else None,
        "categorical_summary": {c: df[c].value_counts() for c in categorical},
    }
    eda["strongest_correlation"] = strongest_correlation(eda["correlation"])

    figures = []
    if eda["correlation"] is not None:
        figures.append(("Correlation heatmap", _heatmap(eda["correlation"])))
    best = eda["strongest_correlation"]
    if best is not None:
        figures.append((f"{best[1]} vs {best[0]}", _scatter(df, best[0], best[1])))
    for col in numeric[:MAX_HISTOGRAMS]:
        figures.append((col, _histogram(df, col)))
    for col in categorical[:MAX_BAR_CHARTS]:
        figures.append((col, _bar_chart(df, col)))
    return eda, figures


# ----------------------------------------------------------------------
# Machine learning
# ----------------------------------------------------------------------
def run_ml(df, target, id_columns):
    """Train a Random Forest regressor and report how well it predicts ``target``."""
    X = df.drop(columns=[target] + list(id_columns))
    y = df[target]

    # drop text columns that are practically unique per row (names, emails ...)
    for col in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[col]) and X[col].nunique() > max(20, 0.5 * len(X)):
            X = X.drop(columns=col)
    if X.shape[1] == 0:
        raise ValueError("no usable feature columns remain after removing identifiers.")

    X = pd.get_dummies(X, drop_first=True).astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    cv_r2 = None
    if len(df) >= 50:
        cv_r2 = float(
            cross_val_score(
                RandomForestRegressor(n_estimators=100, random_state=42),
                X, y, cv=5, scoring="r2",
            ).mean()
        )

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    return {
        "target_column": target,
        "model": "Random Forest Regressor",
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "cv_r2": cv_r2,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_importance": importances,
    }


# ----------------------------------------------------------------------
# Anomaly detection
# ----------------------------------------------------------------------
def run_anomaly_detection(df, numeric):
    """Flag unusual rows with an Isolation Forest. Anomaly == -1 means unusual."""
    contamination = 0.1 if len(df) < 50 else 0.05
    scaled = StandardScaler().fit_transform(df[numeric])
    forest = IsolationForest(contamination=contamination, random_state=42)
    flags = forest.fit_predict(scaled)

    result = df.copy()
    result["Anomaly"] = flags
    result["Anomaly_Score"] = forest.decision_function(scaled).round(4)
    return result, int((flags == -1).sum())


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def execute_agent_plan(df, plan):
    """Run everything the plan asked for. A failure in one step never stops the others."""
    results = {
        "eda": None,
        "figures": [],
        "ml": None,
        "ml_error": None,
        "anomalies": None,
        "anomaly_count": 0,
        "anomaly_error": None,
    }
    numeric = plan.get("numeric_columns", [])
    categorical = plan.get("categorical_columns", [])

    if plan["run_eda"]:
        try:
            results["eda"], results["figures"] = run_eda(df, numeric, categorical)
        except Exception as error:  # keep the app alive
            results["eda_error"] = str(error)

    if plan["run_ml"]:
        try:
            results["ml"] = run_ml(df, plan["target_column"], plan.get("id_columns", []))
        except Exception as error:
            results["ml_error"] = str(error)

    if plan["run_anomaly_detection"]:
        try:
            results["anomalies"], results["anomaly_count"] = run_anomaly_detection(df, numeric)
        except Exception as error:
            results["anomaly_error"] = str(error)

    return results

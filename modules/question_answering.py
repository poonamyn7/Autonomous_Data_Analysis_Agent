"""Step 6 - answer plain-English questions about the dataset.

The agent first classifies the question, picks a tool (a pandas calculation or
a stored analysis result), runs it, and returns both the answer and the
decision it made so the app can show its reasoning.
"""

import re

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from modules.utils import (
    describe_correlation,
    explain_anomaly_record,
    get_id_columns,
    row_label,
    strongest_correlation,
)

MISSING_RE = r"\b(missing|null|nan|empty|blank)\b"
ANOMALY_RE = r"\b(anomal\w*|outlier\w*|unusual|abnormal|suspicious)\b"
CORR_RE = r"\b(correlat\w*|relationship|related|relate|affect\w*|impact\w*|influenc\w*|depend\w*|associat\w*)\b"
ML_RE = r"\b(predict\w*|model|accuracy|r2|mae|error|important|importance|features?|machine learning|ml)\b"
SHAPE_RE = r"\b(rows|records|columns|shape|size|dimensions|fields|variables)\b"
SUMMARY_RE = r"\b(summary|summarize|summarise|describe|overview|statistics|stats)\b"
COUNT_RE = r"\b(how many|count|number of|distribution|frequency|most common|unique)\b"

AGGREGATIONS = [
    ("max", r"\b(max|maximum|highest|largest|biggest|best|top)\b"),
    ("min", r"\b(min|minimum|lowest|smallest|worst|least)\b"),
    ("median", r"\b(median|middle)\b"),
    ("sum", r"\b(sum|total)\b"),
    ("std", r"\b(std|deviation|spread|variation)\b"),
    ("mean", r"\b(avg|average|mean)\b"),
]
AGG_WORDS = {
    "max": "highest", "min": "lowest", "median": "median",
    "sum": "total", "std": "standard deviation of", "mean": "average",
}


def _has(pattern, text):
    return re.search(pattern, text) is not None


def _mentioned(name, question):
    variants = {str(name).lower(), str(name).lower().replace("_", " ")}
    return any(re.search(rf"\b{re.escape(v)}\b", question) for v in variants)


def _detect_columns(question, df):
    return [c for c in df.columns if _mentioned(c, question)]


def _detect_values(question, df, categorical):
    """Find category values named in the question, e.g. 'female' in Gender."""
    found = []
    for col in categorical:
        for value in df[col].dropna().unique():
            if re.search(rf"\b{re.escape(str(value).lower())}\b", question):
                found.append((col, value))
    return found


def _decision(kind, tool, reason, columns=None):
    return {"type": kind, "tool": tool, "reason": reason, "columns": columns or []}


def _format(value):
    if isinstance(value, float):
        return f"{value:,.2f}"
    return f"{value}"


def _record_text(row):
    return ", ".join(f"{k} = {_format(v)}" for k, v in row.items())


def answer_question(question, analysis_results):
    """Return ``(answer_markdown, decision_dict)`` for a question."""
    df = analysis_results["dataframe"]
    profile = analysis_results["profile"]
    plan = analysis_results.get("plan", {})
    q = question.lower().strip().replace("r²", "r2")

    id_columns = get_id_columns(df)
    numeric_all = [
        c for c in df.columns
        if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c]) and c not in id_columns
    ]
    categorical = profile.get("categorical_columns", [])
    mentioned = _detect_columns(q, df)
    mentioned_num = [c for c in mentioned if c in numeric_all]
    mentioned_cat = [c for c in mentioned if c in categorical]
    value_filters = _detect_values(q, df, categorical)

    # ---------------- missing values ----------------
    if _has(MISSING_RE, q):
        before = profile.get("missing_before_cleaning", {})
        total = sum(before.values())
        decision = _decision(
            "Data quality", "Missing-value profile",
            "The question asks about missing or empty data.",
        )
        if total == 0:
            return "No missing values were found in the uploaded data.", decision
        detail = ", ".join(f"{c}: {n}" for c, n in before.items() if n > 0)
        return (
            f"**{total}** missing value(s) were found before cleaning ({detail}). "
            "They were filled automatically (median for numbers, most common value for text).",
            decision,
        )

    # ---------------- anomalies ----------------
    if _has(ANOMALY_RE, q):
        anomalies = analysis_results.get("anomalies", [])
        decision = _decision(
            "Anomaly", "Isolation Forest results",
            "The question asks about unusual or outlier records.",
        )
        if not plan.get("run_anomaly_detection", True):
            return "Anomaly detection was not run for this dataset (not enough rows or numeric columns).", decision
        if not anomalies:
            return "No potential anomalies were detected.", decision
        lines = [f"**{len(anomalies)}** potential anomalies were detected:"]
        for record in anomalies[:10]:
            label = row_label(record, id_columns)
            lines.append(f"- **{label}**: {explain_anomaly_record(record, df, plan.get('numeric_columns', numeric_all))}")
        return "\n".join(lines), decision

    # ---------------- group by ----------------
    if (_has(r"\b(by|per|each|across|grouped)\b", q)) and mentioned_cat and mentioned_num:
        group_col, value_col = mentioned_cat[0], mentioned_num[0]
        agg = "mean"
        for name, pattern in AGGREGATIONS:
            if _has(pattern, q) and name in ("sum", "max", "min", "median"):
                agg = name
                break
        table = df.groupby(group_col)[value_col].agg(agg).round(2)
        lines = [f"{AGG_WORDS.get(agg, 'average').capitalize()} **{value_col}** by **{group_col}**:"]
        lines += [f"- {idx}: **{val:,.2f}**" for idx, val in table.items()]
        return "\n".join(lines), _decision(
            "Group comparison", "pandas groupby",
            f"The question compares {value_col} across groups of {group_col}.",
            [group_col, value_col],
        )

    # ---------------- correlation ----------------
    if _has(CORR_RE, q):
        decision = _decision(
            "Correlation", "Correlation matrix",
            "The question asks how columns relate to each other.", mentioned_num,
        )
        if len(numeric_all) < 2:
            return "At least two numeric columns are needed to measure correlation.", decision
        corr = df[numeric_all].corr()
        if len(mentioned_num) >= 2:
            a, b = mentioned_num[:2]
            r = corr.loc[a, b]
            return (
                f"The correlation between **{a}** and **{b}** is **{r:.3f}** "
                f"({describe_correlation(r)}). Correlation does not prove causation.",
                decision,
            )
        if len(mentioned_num) == 1:
            col = mentioned_num[0]
            others = corr[col].drop(col).dropna().sort_values(key=abs, ascending=False).head(3)
            lines = [f"Columns most strongly related to **{col}**:"]
            lines += [f"- **{n}**: {v:.3f} ({describe_correlation(v)})" for n, v in others.items()]
            return "\n".join(lines), decision
        best = strongest_correlation(corr)
        a, b, r = best
        return (
            f"The strongest relationship in the data is between **{a}** and **{b}** "
            f"(correlation **{r:.3f}**, {describe_correlation(r)}).",
            decision,
        )

    # ---------------- machine learning ----------------
    if _has(ML_RE, q):
        ml = analysis_results.get("ml")
        decision = _decision(
            "Machine learning", "Random Forest results",
            "The question asks about prediction, accuracy or important features.",
        )
        if ml is None:
            reason = plan.get("ml_skip_reason") or "it was not requested"
            return f"Machine Learning was not run because {reason}", decision
        lines = [
            f"The **{ml['model']}** predicts **{ml['target_column']}** with R² = **{ml['r2']:.2f}** "
            f"and MAE = **{ml['mae']:.2f}**.",
            "Most important features:",
        ]
        lines += [f"- **{n}**: {v * 100:.1f}%" for n, v in ml["feature_importance"].head(5).items()]
        return "\n".join(lines), decision

    # ---------------- shape ----------------
    if _has(SHAPE_RE, q) and not mentioned_num:
        decision = _decision("Dataset structure", "Dataset profile", "The question asks about the size or columns of the dataset.")
        if _has(r"\b(what|which|list|name|show)\b", q) and "column" in q:
            return "The columns are: " + ", ".join(f"**{c}**" for c in df.columns) + ".", decision
        return f"The dataset has **{len(df)} rows** and **{df.shape[1]} columns**.", decision

    # ---------------- counting category values ----------------
    if value_filters and _has(COUNT_RE, q):
        mask = pd.Series(True, index=df.index)
        for col, value in value_filters:
            mask &= df[col] == value
        label = " and ".join(f"{c} = {v}" for c, v in value_filters)
        return (
            f"**{int(mask.sum())}** of {len(df)} rows have {label} ({mask.mean() * 100:.1f}%).",
            _decision("Count", "pandas filter", "The question asks how many rows match a category.", [c for c, _ in value_filters]),
        )

    # ---------------- statistics on numeric columns ----------------
    agg = next((name for name, pattern in AGGREGATIONS if _has(pattern, q)), None)
    if mentioned_num or (agg and numeric_all and not mentioned_cat):
        columns = mentioned_num or numeric_all
        subset = df
        for col, value in value_filters:
            subset = subset[subset[col] == value]
        scope = ""
        if value_filters:
            scope = " for " + " and ".join(f"{c} = {v}" for c, v in value_filters)
        if subset.empty:
            return "No rows match that filter.", _decision("Statistic", "pandas", "Filter matched no rows.", columns)
        decision = _decision(
            "Statistic", "pandas aggregation",
            f"The question asks for a {agg or 'summary'} statistic of numeric column(s).", columns,
        )
        lines = []
        for col in columns:
            s = subset[col]
            if agg in ("max", "min"):
                idx = s.idxmax() if agg == "max" else s.idxmin()
                lines.append(
                    f"The {AGG_WORDS[agg]} **{col}**{scope} is **{_format(s.loc[idx])}**. "
                    f"Record: {_record_text(subset.loc[idx])}."
                )
            elif agg:
                value = getattr(s, agg)()
                lines.append(f"The {AGG_WORDS[agg]} **{col}**{scope} is **{_format(float(value))}**.")
            else:
                lines.append(
                    f"**{col}**{scope}: average {s.mean():.2f}, median {s.median():.2f}, "
                    f"min {s.min():.2f}, max {s.max():.2f}, std {s.std():.2f}."
                )
        return "\n\n".join(lines), decision

    # ---------------- category distribution ----------------
    if mentioned_cat or (_has(COUNT_RE, q) and categorical):
        col = (mentioned_cat or categorical)[0]
        counts = df[col].value_counts()
        lines = [f"Distribution of **{col}**:"]
        lines += [f"- {k}: **{v}** ({v / len(df) * 100:.1f}%)" for k, v in counts.items()]
        return "\n".join(lines), _decision(
            "Distribution", "pandas value_counts", f"The question asks about the values in {col}.", [col]
        )

    # ---------------- overall summary ----------------
    if _has(SUMMARY_RE, q) and numeric_all:
        table = df[numeric_all].describe().round(2)
        lines = ["Summary of the numeric columns:"]
        for col in numeric_all:
            lines.append(
                f"- **{col}**: mean {table.loc['mean', col]}, min {table.loc['min', col]}, "
                f"max {table.loc['max', col]}, std {table.loc['std', col]}"
            )
        return "\n".join(lines), _decision("Summary", "pandas describe", "The question asks for an overview of the data.")

    # ---------------- fallback ----------------
    example_num = numeric_all[-1] if numeric_all else "a numeric column"
    return (
        "I could not match that question to an analysis. Try questions like:\n"
        f"- What is the average {example_num}?\n"
        f"- Who has the highest {example_num}?\n"
        "- How many rows are there?\n"
        "- Which columns are most correlated?\n"
        "- Are there any anomalies?\n"
        "- Which features matter most for the prediction?",
        _decision("Unknown", "None", "No known pattern (statistic, correlation, anomaly, ML ...) matched the question."),
    )

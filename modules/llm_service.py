"""Step 5 - turn the numbers into written insights.

This is a rule-based "insight writer": it needs no API key and always works
offline. To use a real LLM later, replace the body of ``generate_insights``
with a call to your model of choice and pass it ``build_facts(...)`` below.
"""

from modules.utils import describe_correlation, explain_anomaly_record, row_label


def _summary_section(df, profile):
    n_num = len(profile["numeric_columns"])
    n_cat = len(profile["categorical_columns"])
    lines = [
        f"- The dataset contains **{profile['rows']} rows** and "
        f"**{profile['columns']} columns** ({n_num} numeric, {n_cat} categorical).",
    ]
    if profile["id_columns"]:
        lines.append(
            f"- Identifier column(s) {', '.join(profile['id_columns'])} were left out of "
            "the statistics because they only label rows."
        )
    return "#### 1. Dataset Summary\n" + "\n".join(lines)


def _quality_section(profile):
    missing = {c: n for c, n in profile["missing_before_cleaning"].items() if n > 0}
    lines = []
    if not missing:
        lines.append("- No missing values were found.")
    else:
        total = sum(missing.values())
        cols = ", ".join(f"{c} ({n})" for c, n in missing.items())
        lines.append(f"- **{total} missing value(s)** were found in: {cols}. They were filled automatically.")
    lines += [f"- {entry}" for entry in profile["cleaning_log"] if "Filled" not in entry]
    return "#### 2. Data Quality\n" + "\n".join(lines)


def _numeric_section(df, profile):
    lines = []
    for col in profile["numeric_columns"][:8]:
        s = df[col]
        line = (
            f"- **{col}** ranges from {s.min():.2f} to {s.max():.2f}, "
            f"with an average of {s.mean():.2f}."
        )
        if len(s) >= 8 and abs(s.skew()) > 1:
            side = "high" if s.skew() > 0 else "low"
            line += f" It is skewed towards {side} values."
        lines.append(line)
    for col in profile["categorical_columns"][:4]:
        shares = df[col].value_counts(normalize=True)
        lines.append(
            f"- **{col}**: the most common value is **{shares.index[0]}** "
            f"({shares.iloc[0] * 100:.1f}% of rows)."
        )
    if not lines:
        lines.append("- No numeric or categorical columns could be summarised.")
    return "#### 3. Patterns in the Data\n" + "\n".join(lines)


def _relationship_section(eda, rows):
    best = eda.get("strongest_correlation") if eda else None
    if best is None:
        return "#### 4. Important Relationship\n- At least two numeric columns are needed to measure relationships."
    a, b, r = best
    lines = [
        f"- The strongest relationship is between **{a}** and **{b}** "
        f"(correlation **{r:.3f}**, a {describe_correlation(r)} relationship).",
        "- Correlation shows that two columns move together, but it does not prove that one causes the other.",
    ]
    if rows < 30:
        lines.append(
            f"- With only {rows} rows this correlation is not statistically reliable; treat it as a hint."
        )
    return "#### 4. Important Relationship\n" + "\n".join(lines)


def _ml_section(ml, plan):
    if ml is None:
        reason = plan.get("ml_skip_reason") or "it was not requested."
        return f"#### 5. Machine Learning\n- Machine Learning was not performed because {reason}"
    r2 = ml["r2"]
    quality = "strong" if r2 >= 0.8 else "moderate" if r2 >= 0.5 else "weak"
    lines = [
        f"- A **{ml['model']}** was trained to predict **{ml['target_column']}** "
        f"using {ml['n_train']} rows and tested on {ml['n_test']} unseen rows.",
        f"- R² = **{r2:.2f}** ({quality} fit) and average error (MAE) = **{ml['mae']:.2f}**.",
    ]
    if ml.get("cv_r2") is not None:
        lines.append(f"- 5-fold cross-validated R² = {ml['cv_r2']:.2f}, a more stable estimate than a single split.")
    top = ml["feature_importance"].head(3)
    if len(top):
        names = ", ".join(f"**{n}** ({v * 100:.0f}%)" for n, v in top.items())
        lines.append(f"- The most influential inputs are {names}.")
    return "#### 5. Machine Learning\n" + "\n".join(lines)


def _anomaly_section(df, profile, anomalies):
    count = len(anomalies)
    if count == 0:
        return "#### 6. Potential Anomalies\n- No potential anomalies were detected."
    lines = [
        f"- The agent detected **{count} potential anomalies**. They have unusual numerical "
        "patterns, but an anomaly is not automatically an error.",
    ]
    for record in anomalies[:5]:
        label = row_label(record, profile["id_columns"])
        lines.append(f"- **{label}**: {explain_anomaly_record(record, df, profile['numeric_columns'])}.")
    if count > 5:
        lines.append(f"- ...and {count - 5} more (see the Anomalies tab).")
    return "#### 6. Potential Anomalies\n" + "\n".join(lines)


def _conclusion_section(profile, ml, anomalies):
    missing_total = sum(profile["missing_before_cleaning"].values())
    quality = "good" if missing_total == 0 else "acceptable after cleaning"
    parts = [
        f"The dataset has {quality} basic data quality.",
        f"The agent analysed {len(profile['numeric_columns'])} numeric variable(s).",
    ]
    if ml is not None:
        parts.append(f"The prediction model reached R² = {ml['r2']:.2f}.")
    if anomalies:
        parts.append(f"{len(anomalies)} potential anomalous record(s) should be reviewed.")
    return "#### 7. Overall Conclusion\n- " + " ".join(parts)


def generate_insights(analysis_results):
    """Return a markdown string with the agent's written findings."""
    df = analysis_results["dataframe"]
    profile = analysis_results["profile"]
    eda = analysis_results.get("eda")
    ml = analysis_results.get("ml")
    plan = analysis_results.get("plan", {})
    anomalies = analysis_results.get("anomalies", [])

    sections = [
        _summary_section(df, profile),
        _quality_section(profile),
        _numeric_section(df, profile),
        _relationship_section(eda, profile["rows"]),
        _ml_section(ml, plan),
        _anomaly_section(df, profile, anomalies),
        _conclusion_section(profile, ml, anomalies),
    ]
    return "\n\n".join(sections)

"""Markdown version of the analysis report (also handy for GitHub / notes)."""

from datetime import datetime


def _table(df, max_cols=8):
    df = df.iloc[:, :max_cols].round(2)
    header = "| Statistic | " + " | ".join(map(str, df.columns)) + " |"
    divider = "|---|" + "---|" * len(df.columns)
    rows = [
        f"| {idx} | " + " | ".join(str(v) for v in row) + " |"
        for idx, row in df.iterrows()
    ]
    return "\n".join([header, divider] + rows)


def generate_report(profile, plan, eda_results, ml_results, anomaly_count, insights, warnings=None):
    """Return the full report as a markdown string."""
    lines = [
        "# Autonomous Data Analysis Agent - Report",
        f"_Generated on {datetime.now():%d %B %Y, %H:%M}_",
        "",
        "## 1. Dataset Profile",
        f"- Rows: {profile['rows']}",
        f"- Columns: {profile['columns']}",
        f"- Column names: {', '.join(profile['column_names'])}",
        f"- Missing values before cleaning: {sum(profile['missing_before_cleaning'].values())}",
        "",
        "## 2. AI Agent Decision",
        f"- EDA: {'Run' if plan['run_eda'] else 'Skipped'}",
        f"- Machine Learning: {'Run' if plan['run_ml'] else 'Skipped'}",
        f"- Anomaly Detection: {'Run' if plan['run_anomaly_detection'] else 'Skipped'}",
        f"- Target column: {plan['target_column'] or 'None'}",
        "",
        "Reasons:",
    ]
    lines += [f"- {reason}" for reason in plan["reasons"]]

    lines += ["", "## 3. Exploratory Data Analysis"]
    if eda_results is not None and not eda_results["summary_statistics"].empty:
        lines.append(_table(eda_results["summary_statistics"]))
    else:
        lines.append("EDA was not performed.")

    lines += ["", "## 4. Machine Learning"]
    if ml_results is None:
        lines.append("Machine Learning was not performed.")
    else:
        lines += [
            f"- Target: {ml_results['target_column']}",
            f"- Model: {ml_results['model']}",
            f"- MAE: {ml_results['mae']:.2f}",
            f"- R²: {ml_results['r2']:.2f}",
        ]

    lines += ["", "## 5. Anomaly Detection", f"Potential anomalies detected: {anomaly_count}"]
    lines += ["", "## 6. AI Insights", insights]

    if warnings:
        lines += ["", "## 7. Validation Warnings"] + [f"- {w}" for w in warnings]
    return "\n".join(lines)

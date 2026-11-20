"""Runs the whole agent once and returns everything the UI needs.

Keeping this separate from app.py means the analysis can be tested without
Streamlit, and Streamlit can cache the result so typing a question does not
re-run the whole analysis.
"""

from io import BytesIO

from modules.agent import analyze_dataset
from modules.agent_executor import execute_agent_plan
from modules.data_cleaner import clean_data
from modules.data_profiler import profile_data
from modules.llm_service import generate_insights
from modules.pdf_report_generator import generate_pdf_report
from modules.report_generator import generate_report
from modules.result_validator import validate_results


def run_pipeline(raw_df, target_override=None):
    df = clean_data(raw_df)
    profile = profile_data(df)
    plan = analyze_dataset(df, target_override)
    execution = execute_agent_plan(df, plan)

    anomaly_df = execution["anomalies"]
    anomaly_count = execution["anomaly_count"]
    anomaly_records = []
    if anomaly_df is not None and anomaly_count > 0:
        flagged = anomaly_df[anomaly_df["Anomaly"] == -1].copy()
        flagged["_row"] = flagged.index + 1
        anomaly_records = flagged.to_dict(orient="records")

    analysis_results = {
        "dataframe": df,
        "profile": profile,
        "plan": plan,
        "eda": execution["eda"],
        "ml": execution["ml"],
        "anomalies": anomaly_records,
    }
    insights = generate_insights(analysis_results)
    validation = validate_results(profile, execution["eda"], execution["ml"], anomaly_count, plan)

    pdf_buffer = BytesIO()
    generate_pdf_report(
        pdf_buffer, profile, plan, execution["eda"], execution["ml"], anomaly_count,
        insights, figures=execution["figures"], warnings=validation["warnings"],
        anomaly_records=anomaly_records,
    )
    markdown = generate_report(
        profile, plan, execution["eda"], execution["ml"], anomaly_count,
        insights, warnings=validation["warnings"],
    )

    return {
        "df": df,
        "profile": profile,
        "plan": plan,
        "execution": execution,
        "analysis_results": analysis_results,
        "insights": insights,
        "validation": validation,
        "pdf_bytes": pdf_buffer.getvalue(),
        "markdown_report": markdown,
    }

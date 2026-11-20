"""Autonomous Data Analysis Agent - Streamlit app.

Run with:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from modules.file_loader import (
    combine_datasets,
    group_by_schema,
    load_folder,
    load_uploaded_files,
    summarize,
)
from modules.pipeline import run_pipeline
from modules.question_answering import answer_question

BASE_DIR = Path(__file__).parent
DEFAULT_DATASETS = [
    BASE_DIR / "data" / "students.csv",
    BASE_DIR / "data" / "sample_students_large.csv",
]
AUTO_TARGET = "Auto (agent decides)"

st.set_page_config(page_title="Autonomous Data Analysis Agent", page_icon="🤖", layout="wide")


# ==========================================================
# CACHED ANALYSIS - runs once per dataset, not on every click
# ==========================================================
@st.cache_resource(show_spinner="The agent is analysing your data...")
def cached_pipeline(raw_df, target_override=None):
    return run_pipeline(raw_df, target_override)


# ==========================================================
# SIDEBAR: DATA SOURCE
# ==========================================================
st.title("🤖 Autonomous Data Analysis Agent")
st.caption("Upload one or many CSV files (or a ZIP of folders) and the agent decides what to analyse, "
           "runs it, explains it and writes a report.")

with st.sidebar:
    st.header("Dataset")
    uploaded_files = st.file_uploader(
        "Upload CSV files or a ZIP",
        type=["csv", "zip"],
        accept_multiple_files=True,
        help="Select many CSV files at once (Ctrl+A inside the folder), or upload a ZIP of a "
             "whole folder - sub-folders are searched automatically.",
    )
    folder_path = st.text_input(
        "...or type a folder path on this computer",
        placeholder=r"G:\My Drive\data",
        help="Every CSV in this folder and all of its sub-folders is loaded. "
             "Works when the app runs on your own computer.",
    )


def load_datasets():
    """Return (datasets_dict, source_label) or stop the app with a friendly message."""
    datasets, problems = {}, []
    if uploaded_files:
        found, issues = load_uploaded_files(uploaded_files)
        datasets.update(found)
        problems += issues
    if folder_path.strip():
        found, issues = load_folder(folder_path)
        for name, frame in found.items():
            datasets[name if name not in datasets else f"{folder_path.strip()}/{name}"] = frame
        problems += issues

    for message in problems:
        st.sidebar.warning(message)

    if datasets:
        return datasets, f"{len(datasets)} file(s) loaded"

    if uploaded_files or folder_path.strip():
        st.error("None of the selected files could be used. See the warnings in the sidebar.")
        st.stop()

    path = next((p for p in DEFAULT_DATASETS if p.exists()), None)
    if path is None:
        st.info("Upload CSV files in the sidebar to begin.")
        st.stop()
    return {path.name: pd.read_csv(path)}, f"{path.name} (default dataset)"


datasets, source_label = load_datasets()

if len(datasets) == 1:
    source_name, raw_df = next(iter(datasets.items()))
    combine_notes = []
else:
    with st.sidebar:
        st.caption(f"{len(datasets)} files loaded")
        with st.expander("Loaded files"):
            st.dataframe(summarize(datasets)[["File", "Rows", "Columns"]])
        groups = group_by_schema(datasets)
        same_columns = len(groups) == 1
        modes = [
            "Combine all files into one dataset",
            "Combine files that have the same columns",
            "Analyse one file",
        ]
        mode = st.radio(
            "How should the agent use these files?",
            modes,
            index=0 if same_columns else 1,
            help="Combine stacks the files and adds a Source_File column. Use 'same columns' when "
                 "the folder mixes different kinds of files, or analyse a single file.",
        )
        if mode == modes[2]:
            source_name = st.selectbox("File to analyse", list(datasets))
            raw_df, combine_notes = datasets[source_name], []
        elif mode == modes[1]:
            label = st.selectbox("Group of files", list(groups))
            raw_df, combine_notes = combine_datasets(groups[label])
            source_name = f"{len(groups[label])} files combined"
        else:
            raw_df, combine_notes = combine_datasets(datasets)
            source_name = f"{len(datasets)} files combined"
    for note in combine_notes:
        st.warning(note)

try:
    result = cached_pipeline(raw_df)
except Exception as error:
    st.error(f"The agent could not analyse this file: {error}")
    st.stop()

with st.sidebar:
    st.caption(f"Using: {source_name}")
    options = [AUTO_TARGET] + result["plan"]["numeric_columns"]
    choice = st.selectbox(
        "Target column for Machine Learning",
        options,
        help="Leave on Auto and the agent picks one, or choose the column you want to predict.",
    )

if choice != AUTO_TARGET:
    result = cached_pipeline(raw_df, choice)

df = result["df"]
profile = result["profile"]
plan = result["plan"]
execution = result["execution"]
analysis_results = result["analysis_results"]
eda_results = execution["eda"]
ml_results = execution["ml"]
anomaly_df = execution["anomalies"]
anomaly_count = execution["anomaly_count"]

tabs = st.tabs([
    "📋 Overview", "📊 EDA", "🧠 Machine Learning",
    "🚨 Anomalies", "💡 AI Insights", "💬 Ask AI", "📄 Report",
])
tab_overview, tab_eda, tab_ml, tab_anomaly, tab_insights, tab_ask, tab_report = tabs


# ==========================================================
# OVERVIEW: agent decision + dataset + profile + validation
# ==========================================================
with tab_overview:
    st.subheader("🤖 AI Agent Decision")
    st.write("The agent automatically decided:")

    c1, c2, c3 = st.columns(3)
    for column, label, flag in [
        (c1, "EDA", plan["run_eda"]),
        (c2, "Machine Learning", plan["run_ml"]),
        (c3, "Anomaly Detection", plan["run_anomaly_detection"]),
    ]:
        with column:
            if flag:
                st.success(f"{label} → Run")
            else:
                st.info(f"{label} → Skip")

    if plan["target_column"]:
        st.write("**Selected Target:**", plan["target_column"])

    st.write("**Why did the Agent make these decisions?**")
    for reason in plan["reasons"]:
        st.info(reason)

    st.subheader("Dataset")
    st.dataframe(df)

    st.subheader("Dataset Profile")
    left, right = st.columns(2)
    with left:
        st.metric("Rows", profile["rows"])
        st.metric("Columns", profile["columns"])
    with right:
        st.write("**Column Names**")
        st.write(profile["column_names"])

    st.write("**Columns at a glance**")
    glance = pd.DataFrame({
        "Data type": profile["data_types"],
        "Missing (before cleaning)": profile["missing_before_cleaning"],
    })
    st.dataframe(glance)

    if profile["cleaning_log"]:
        st.write("**What the agent cleaned automatically**")
        for entry in profile["cleaning_log"]:
            st.write(f"- {entry}")

    st.subheader("Result Validation")
    if result["validation"]["warnings"]:
        for warning in result["validation"]["warnings"]:
            st.warning(warning)
    else:
        st.success("Analysis results passed validation.")


# ==========================================================
# EDA + VISUALISATIONS
# ==========================================================
with tab_eda:
    if eda_results is None:
        st.info("EDA was skipped for this dataset.")
    else:
        st.subheader("Automatic EDA")
        st.write("### Summary Statistics")
        st.dataframe(eda_results["summary_statistics"])

        st.write("### Correlation")
        if eda_results["correlation"] is not None:
            st.dataframe(eda_results["correlation"])
        else:
            st.write("Not enough numerical columns for correlation analysis.")

        if eda_results["categorical_summary"]:
            st.write("### Categorical Columns")
            for column, counts in eda_results["categorical_summary"].items():
                st.write(f"**{column}**")
                st.dataframe(counts.rename("Count"))

    figures = execution["figures"]
    if figures:
        st.subheader("Automatic Visualizations")
        for i in range(0, len(figures), 2):
            for column, (title, figure) in zip(st.columns(2), figures[i:i + 2]):
                with column:
                    st.pyplot(figure)


# ==========================================================
# MACHINE LEARNING
# ==========================================================
with tab_ml:
    if execution["ml_error"]:
        st.warning(f"Machine Learning could not run: {execution['ml_error']}")

    if ml_results is not None:
        st.subheader("Machine Learning Results")
        st.write("Target Column:", ml_results["target_column"])
        st.caption(
            f"{ml_results['model']} · trained on {ml_results['n_train']} rows, "
            f"tested on {ml_results['n_test']} unseen rows"
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Absolute Error (MAE)", round(ml_results["mae"], 2))
        m2.metric("R² Score", round(ml_results["r2"], 2))
        if ml_results["cv_r2"] is not None:
            m3.metric("Cross-validated R²", round(ml_results["cv_r2"], 2))

        st.write("**Which columns matter most for the prediction?**")
        st.bar_chart(ml_results["feature_importance"])
    else:
        st.info(
            "Machine Learning was not run. "
            + (plan.get("ml_skip_reason") or "")
        )


# ==========================================================
# ANOMALY DETECTION
# ==========================================================
with tab_anomaly:
    if execution.get("anomaly_error"):
        st.warning(f"Anomaly detection could not run: {execution['anomaly_error']}")

    if anomaly_df is not None:
        st.subheader("Anomaly Detection")
        st.write("Potential anomalies detected:", anomaly_count)

        if anomaly_count > 0:
            st.write("### Potential Anomaly Records")
            st.dataframe(anomaly_df[anomaly_df["Anomaly"] == -1])
            st.info(
                "These records are marked as potential anomalies because their numerical "
                "patterns are unusual compared with the rest of the dataset. "
                "An anomaly does not necessarily mean the record is incorrect."
            )
        else:
            st.success("No potential anomalies were detected.")
    else:
        st.info("Anomaly detection was skipped for this dataset.")


# ==========================================================
# AI INSIGHTS
# ==========================================================
with tab_insights:
    st.subheader("AI Insights")
    st.markdown(result["insights"])


# ==========================================================
# ASK AI
# ==========================================================
with tab_ask:
    st.subheader("💬 Ask AI About Your Data")
    st.caption("Examples: average marks · who has the highest attendance · average marks by gender · "
               "are there anomalies · which features matter most")
    question = st.text_input("Ask a question about your dataset:")

    if question:
        answer, decision = answer_question(question, analysis_results)

        st.write("### 🤖 Agent Decision")
        st.info(
            f"**Question Type:** {decision['type']}\n\n"
            f"**Tool Selected:** {decision['tool']}\n\n"
            f"**Columns Detected:** {decision.get('columns', [])}\n\n"
            f"**Reason:** {decision['reason']}"
        )

        st.write("### 💬 AI Answer")
        st.markdown(answer)


# ==========================================================
# REPORT DOWNLOADS
# ==========================================================
with tab_report:
    st.subheader("📄 Automatic Analysis Report")
    st.write("The report contains the profile, agent decisions, statistics, charts, "
             "model results, anomalies, insights and validation warnings.")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⬇️ Download PDF Report",
            data=result["pdf_bytes"],
            file_name="autonomous_data_analysis_report.pdf",
            mime="application/pdf",
        )
    with d2:
        st.download_button(
            label="⬇️ Download Markdown Report",
            data=result["markdown_report"],
            file_name="autonomous_data_analysis_report.md",
            mime="text/markdown",
        )

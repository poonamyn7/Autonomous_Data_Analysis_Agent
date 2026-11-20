"""Step 8 - build the downloadable PDF report with ReportLab."""

import re
from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    KeepTogether,
    Table,
    TableStyle,
)

MAX_PDF_FIGURES = 6
MAX_TABLE_COLUMNS = 7


def _md_inline(text):
    """Escape text for ReportLab and turn **bold** into <b>bold</b>."""
    text = escape(text).replace("²", "<super>2</super>")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("T", parent=base["Title"], fontSize=22, spaceAfter=4),
        "subtitle": ParagraphStyle("S", parent=base["Normal"], fontSize=11,
                                   textColor=colors.grey, alignment=1, spaceAfter=14),
        "h1": ParagraphStyle("H1", parent=base["Heading2"], textColor=colors.HexColor("#1F4FBF"),
                             spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("H2", parent=base["Heading4"], spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("B", parent=base["BodyText"], fontSize=10, leading=14),
        "bullet": ParagraphStyle("BL", parent=base["BodyText"], fontSize=10, leading=14,
                                 leftIndent=14, bulletIndent=4),
        "warn": ParagraphStyle("W", parent=base["BodyText"], fontSize=10, leading=14,
                               leftIndent=14, bulletIndent=4, textColor=colors.HexColor("#9A5B00")),
        "small": ParagraphStyle("SM", parent=base["BodyText"], fontSize=9, textColor=colors.grey),
    }


def _data_table(frame, round_to=2):
    """A compact, styled table from a DataFrame (index becomes first column)."""
    frame = frame.iloc[:, :MAX_TABLE_COLUMNS].round(round_to)
    data = [[str(frame.index.name or "")] + [str(c) for c in frame.columns]]
    for idx, row in frame.iterrows():
        data.append([str(idx)] + [f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in row])
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4FBF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6FC")]),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]))
    return table


def _figure_image(fig, width_cm):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130)
    buffer.seek(0)
    width = width_cm * cm
    height = width * fig.get_figheight() / fig.get_figwidth()
    return Image(buffer, width=width, height=height)


def _figure_grid(figures):
    """Two figures per row."""
    images = [_figure_image(fig, 8.2) for _, fig in figures[:MAX_PDF_FIGURES]]
    rows = [images[i:i + 2] for i in range(0, len(images), 2)]
    for row in rows:
        while len(row) < 2:
            row.append("")
    table = Table(rows)
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return table


def _insights_flowables(insights, styles):
    items = []
    for line in insights.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("####"):
            items.append(Paragraph(_md_inline(line.lstrip("# ").strip()), styles["h2"]))
        elif line.startswith("- "):
            items.append(Paragraph(_md_inline(line[2:]), styles["bullet"], bulletText="•"))
        else:
            items.append(Paragraph(_md_inline(line), styles["body"]))
    return items


def generate_pdf_report(
    output,
    profile,
    plan,
    eda_results,
    ml_results,
    anomaly_count,
    insights,
    figures=None,
    warnings=None,
    anomaly_records=None,
):
    """Write the report to ``output`` (a file path or a BytesIO object)."""
    styles = _styles()
    doc = SimpleDocTemplate(
        output, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm, title="Autonomous Data Analysis Report",
    )
    story = [
        Paragraph("Autonomous Data Analysis Agent", styles["title"]),
        Paragraph(f"Automatic Dataset Analysis Report &middot; {datetime.now():%d %b %Y}", styles["subtitle"]),
    ]

    # 1. profile
    missing_total = sum(profile.get("missing_before_cleaning", {}).values())
    story += [
        Paragraph("1. Dataset Profile", styles["h1"]),
        Paragraph(f"<b>Rows:</b> {profile['rows']} &nbsp;&nbsp; <b>Columns:</b> {profile['columns']}", styles["body"]),
        Paragraph(f"<b>Column names:</b> {_md_inline(', '.join(profile['column_names']))}", styles["body"]),
        Paragraph(f"<b>Missing values (before cleaning):</b> {missing_total}", styles["body"]),
    ]
    for entry in profile.get("cleaning_log", []):
        story.append(Paragraph(_md_inline(entry), styles["bullet"], bulletText="•"))

    # 2. decisions
    def state(flag):
        return "Run" if flag else "Skipped"

    story += [
        Paragraph("2. AI Agent Decision", styles["h1"]),
        Paragraph(f"<b>EDA:</b> {state(plan['run_eda'])}", styles["body"]),
        Paragraph(f"<b>Machine Learning:</b> {state(plan['run_ml'])}", styles["body"]),
        Paragraph(f"<b>Anomaly Detection:</b> {state(plan['run_anomaly_detection'])}", styles["body"]),
        Paragraph(f"<b>Target column:</b> {_md_inline(str(plan['target_column'] or 'None'))}", styles["body"]),
        Spacer(1, 4),
    ]
    for reason in plan.get("reasons", []):
        story.append(Paragraph(_md_inline(reason), styles["bullet"], bulletText="•"))

    # 3. EDA
    story.append(Paragraph("3. Exploratory Data Analysis", styles["h1"]))
    if eda_results is not None and not eda_results["summary_statistics"].empty:
        story.append(_data_table(eda_results["summary_statistics"]))
        corr = eda_results.get("correlation")
        if corr is not None:
            story += [Spacer(1, 8), Paragraph("Correlation matrix", styles["h2"]), _data_table(corr)]
    else:
        story.append(Paragraph("EDA was not performed.", styles["body"]))

    if figures:
        story += [Spacer(1, 8), KeepTogether([Paragraph("Charts", styles["h2"]), _figure_grid(figures)])]

    # 4. ML
    story.append(Paragraph("4. Machine Learning", styles["h1"]))
    if ml_results is None:
        reason = plan.get("ml_skip_reason")
        text = "Machine Learning was not performed."
        if reason:
            text += f" Reason: {reason}"
        story.append(Paragraph(_md_inline(text), styles["body"]))
    else:
        story += [
            Paragraph(f"<b>Target:</b> {_md_inline(ml_results['target_column'])} &nbsp;&nbsp; "
                      f"<b>Model:</b> {ml_results['model']}", styles["body"]),
            Paragraph(f"<b>MAE:</b> {ml_results['mae']:.2f} &nbsp;&nbsp; <b>R<super>2</super>:</b> {ml_results['r2']:.2f}", styles["body"]),
            Paragraph("Top features", styles["h2"]),
        ]
        for name, value in ml_results["feature_importance"].head(5).items():
            story.append(Paragraph(f"<b>{_md_inline(str(name))}</b>: {value * 100:.1f}%", styles["bullet"], bulletText="•"))

    # 5. anomalies
    story += [
        Paragraph("5. Anomaly Detection", styles["h1"]),
        Paragraph(f"Potential anomalies detected: <b>{anomaly_count}</b>", styles["body"]),
        Paragraph("An anomaly is a potentially unusual record and does not automatically mean "
                  "that the record is incorrect.", styles["small"]),
    ]
    if anomaly_records:
        import pandas as pd
        frame = pd.DataFrame(anomaly_records).drop(columns=["_row", "Anomaly", "Anomaly_Score"], errors="ignore")
        frame = frame.head(10).set_index(frame.columns[0])
        story += [Spacer(1, 6), _data_table(frame)]

    # 6. insights
    story.append(Paragraph("6. AI Insights", styles["h1"]))
    story += _insights_flowables(insights, styles)

    # 7. validation
    if warnings:
        story.append(Paragraph("7. Validation Warnings", styles["h1"]))
        for warning in warnings:
            story.append(Paragraph(_md_inline(warning), styles["warn"], bulletText="•"))

    doc.build(story)

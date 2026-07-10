from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def generate_report(
    filename,
    df,
    problem,
    results,
    insights
):

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(filename)

    elements = []

    elements.append(
        Paragraph(
            "<b>AI Data Analyst Platform Report</b>",
            styles["Title"]
        )
    )

    elements.append(
        Paragraph(
            f"Rows: {df.shape[0]}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Columns: {df.shape[1]}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Problem Type: {problem}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            "<b>Model Performance</b>",
            styles["Heading2"]
        )
    )

    for _, row in results.iterrows():

        elements.append(
            Paragraph(
                f"{row.iloc[0]} : {row.iloc[1]}",
                styles["Normal"]
            )
        )

    elements.append(
        Paragraph(
            "<b>AI Insights</b>",
            styles["Heading2"]
        )
    )

    for item in insights:

        elements.append(
            Paragraph(
                item,
                styles["Normal"]
            )
        )

    doc.build(elements)
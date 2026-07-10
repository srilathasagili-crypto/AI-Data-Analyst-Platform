import pandas as pd


def generate_ai_insights(df):

    insights = []

    insights.append(f"Total Rows: {df.shape[0]}")
    insights.append(f"Total Columns: {df.shape[1]}")
    insights.append(f"Missing Values: {df.isnull().sum().sum()}")
    insights.append(f"Duplicate Rows: {df.duplicated().sum()}")

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:

        insights.append(
            f"{col} → Mean: {df[col].mean():.2f}, Min: {df[col].min()}, Max: {df[col].max()}"
        )

    return insights
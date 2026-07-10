import matplotlib.pyplot as plt


def correlation_heatmap(df, numeric_cols):

    if len(numeric_cols) > 1:

        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))

        cax = ax.imshow(
            corr,
            cmap="coolwarm"
        )

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))

        ax.set_xticklabels(
            corr.columns,
            rotation=90
        )

        ax.set_yticklabels(
            corr.columns
        )

        plt.colorbar(cax)

        return fig

    return None



def histogram(df, column):

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(
        df[column].dropna(),
        bins=20
    )

    ax.set_title(column)

    return fig



def boxplot(df, column):

    fig, ax = plt.subplots(figsize=(6, 2))

    ax.boxplot(
        df[column].dropna(),
        vert=False
    )

    ax.set_title(column)

    return fig



def categorical_chart(df, column):

    counts = df[column].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))

    counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(column)

    return fig



def custom_chart(df, chart_type, x_column, y_column):

    fig, ax = plt.subplots(figsize=(7, 5))


    if chart_type == "Bar Chart":

        df.groupby(x_column)[y_column].mean().plot(
            kind="bar",
            ax=ax
        )


    elif chart_type == "Line Chart":

        df.groupby(x_column)[y_column].mean().plot(
            kind="line",
            ax=ax
        )


    elif chart_type == "Scatter Plot":

        ax.scatter(
            df[x_column],
            df[y_column]
        )


    elif chart_type == "Histogram":

        ax.hist(
            df[y_column].dropna(),
            bins=20
        )


    elif chart_type == "Box Plot":

        ax.boxplot(
            df[y_column].dropna(),
            vert=False
        )


    elif chart_type == "Pie Chart":

        df[x_column].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )


    ax.set_title(chart_type)

    return fig



# Main visualization function for app.py

def create_charts(df):

    charts = {}

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns


    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns


    # Correlation chart

    heatmap = correlation_heatmap(
        df,
        numeric_cols
    )

    if heatmap:
        charts["Correlation Heatmap"] = heatmap


    # Numerical columns charts

    for col in numeric_cols:

        charts[f"{col} Histogram"] = histogram(
            df,
            col
        )

        charts[f"{col} Boxplot"] = boxplot(
            df,
            col
        )


    # Categorical charts

    for col in categorical_cols:

        charts[f"{col} Count"] = categorical_chart(
            df,
            col
        )


    return charts
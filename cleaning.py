import pandas as pd


def handle_missing_values(df, option):

    df = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns


    if option == "Remove Rows":

        df = df.dropna()


    elif option == "Fill Numerical Columns with Mean":

        for col in numeric_cols:
            df[col] = df[col].fillna(
                df[col].mean()
            )


    elif option == "Fill Categorical Columns with Mode":

        for col in categorical_cols:

            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(
                    df[col].mode()[0]
                )


    return df



def detect_outliers(df):

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns

    outlier_summary = []


    for col in numeric_cols:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr


        count = (
            (df[col] < lower) |
            (df[col] > upper)
        ).sum()


        outlier_summary.append(
            {
                "Column": col,
                "Outliers": count
            }
        )


    return pd.DataFrame(outlier_summary)



def remove_duplicates(df):

    return df.drop_duplicates()



def data_quality_score(df):

    total = df.shape[0] * df.shape[1]

    missing = df.isnull().sum().sum()


    completeness = (
        (total - missing) / total
    ) * 100


    duplicate_score = (
        (df.shape[0] - df.duplicated().sum())
        / df.shape[0]
    ) * 100


    overall = (
        completeness + duplicate_score
    ) / 2


    return completeness, duplicate_score, overall



# Main cleaning function for app.py

def clean_data(df):

    df = df.copy()


    # Remove duplicate rows

    df = remove_duplicates(df)


    # Fill missing values automatically

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns


    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns


    for col in numeric_cols:

        df[col] = df[col].fillna(
            df[col].mean()
        )


    for col in categorical_cols:

        if df[col].isnull().sum() > 0:

            df[col] = df[col].fillna(
                df[col].mode()[0]
            )


    return df
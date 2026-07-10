import pandas as pd


def load_data(uploaded_file):
    """
    Loads CSV or Excel file
    and returns a pandas DataFrame.
    """

    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    else:
        return None
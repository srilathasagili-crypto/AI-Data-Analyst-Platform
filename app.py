import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from automl import run_automl
from cleaning import clean_data
from visualizations import show_visualizations
from evaluation import evaluate_model
from feature_importance import feature_importance
from ai_insights import generate_ai_insights
from report_generator import generate_pdf_report
from data_loader import load_data

st.set_page_config(page_title="AI Data Analyst Platform", page_icon="📊", layout="wide")
st.title("📊 AI Data Analyst Platform")
st.write("Analyze your business data with AI.")

uploaded_file = st.file_uploader("📂 Upload a CSV or Excel file", type=["csv","xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File uploaded successfully!")

    st.header("🧹 Data Cleaning")
    st.subheader("Missing Value Summary")
    st.dataframe(df.isnull().sum())

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include=["object","category"]).columns

    option = st.selectbox(
        "Choose method for missing values",
        ["Keep Missing Values","Remove Rows",
         "Fill Numerical Columns with Mean",
         "Fill Categorical Columns with Mode"]
    )

    if option == "Remove Rows":
        df = df.dropna()
    elif option == "Fill Numerical Columns with Mean":
        for c in numeric_cols:
            df[c] = df[c].fillna(df[c].mean())
    elif option == "Fill Categorical Columns with Mode":
        for c in categorical_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].mode()[0])

    st.header("🚨 Outlier Detection")
    out = []
    for c in numeric_cols:
        q1,q3 = df[c].quantile([0.25,0.75])
        iqr = q3-q1
        n=((df[c]<q1-1.5*iqr)|(df[c]>q3+1.5*iqr)).sum()
        out.append({"Column":c,"Outliers":n})
    st.dataframe(pd.DataFrame(out))

    st.header("🔁 Duplicate Data Check")
    dup=df.duplicated().sum()
    st.metric("Duplicate Rows",dup)
    if dup>0 and st.checkbox("Remove Duplicate Rows"):
        df=df.drop_duplicates()

    st.header("📊 Data Quality Score")
    total=df.shape[0]*df.shape[1]
    miss=df.isna().sum().sum()
    comp=(total-miss)/total*100 if total else 100
    dscore=(df.shape[0]-df.duplicated().sum())/df.shape[0]*100 if len(df) else 100
    st.write(f"Completeness: {comp:.2f}%")
    st.write(f"Duplicate Score: {dscore:.2f}%")
    st.write(f"Overall Score: {(comp+dscore)/2:.2f}%")

    st.header("✅ Cleaned Dataset")
    st.dataframe(df.head())

    numeric_cols=df.select_dtypes(include="number").columns
    categorical_cols=df.select_dtypes(include=["object","category"]).columns

    st.subheader("📊 Dataset Summary")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Rows",len(df))
    c2.metric("Columns",df.shape[1])
    c3.metric("Missing",df.isna().sum().sum())
    c4.metric("Duplicates",df.duplicated().sum())

    st.subheader("🧾 Column Information")
    st.dataframe(pd.DataFrame({
        "Column":df.columns,
        "Type":df.dtypes.astype(str),
        "Missing":df.isna().sum().values
    }))

    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(include="all"))

    if len(numeric_cols)>1:
        st.subheader("🔥 Correlation Heatmap")
        corr=df[numeric_cols].corr()
        fig,ax=plt.subplots(figsize=(8,6))
        im=ax.imshow(corr,cmap="coolwarm")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns,rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        plt.colorbar(im)
        st.pyplot(fig)

    st.subheader("📉 Histograms")
    for c in numeric_cols:
        fig,ax=plt.subplots()
        ax.hist(df[c].dropna(),bins=20)
        ax.set_title(c)
        st.pyplot(fig)

    st.subheader("📦 Box Plots")
    for c in numeric_cols:
        fig,ax=plt.subplots()
        ax.boxplot(df[c].dropna(),vert=False)
        ax.set_title(c)
        st.pyplot(fig)

    st.subheader("📊 Categorical Analysis")
    for c in categorical_cols:
        st.write(c)
        counts=df[c].value_counts()
        st.dataframe(counts)
        fig,ax=plt.subplots()
        counts.plot(kind="bar",ax=ax)
        st.pyplot(fig)

    st.subheader("🔢 Unique Values")
    st.dataframe(pd.DataFrame({"Column":df.columns,"Unique":df.nunique().values}))

    csv=df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Cleaned Dataset",csv,"cleaned_dataset.csv","text/csv")

    # INTERACTIVE VISUALIZATION #

    st.header("📊 Interactive Visualization Dashboard")

    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Bar Chart",
            "Line Chart",
            "Scatter Plot",
            "Histogram",
            "Box Plot",
            "Pie Chart"
        ]
    )
  
    x_column = st.selectbox(
        "Select X-axis Column",
        df.columns
    )

    y_column = st.selectbox(
        "Select Y-axis Column",
        numeric_cols
    )

    fig, ax = plt.subplots(figsize=(7,5))

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

    st.pyplot(fig)

    chart_name = chart_type.lower().replace(" ", "_") + ".png"

    fig.savefig(chart_name)

    with open(chart_name, "rb") as file:

        st.download_button(
            "⬇️ Download Chart",
            file,
            file_name=chart_name,
            mime="image/png"
        )

    # AutoML #

    st.subheader("🤖 AutoML Model Training")

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    if st.button("🚀 Run AutoML"):

        problem, results, best_model, X_test, y_test, feature_names = run_automl(
            df,
            target_column
        )

        st.success(
            f"Problem Type: {problem}"
        )

        st.write("### 📊 Model Performance")

        st.dataframe(results)


        best_row = results.iloc[
            results.iloc[:, 1].idxmax()
        ]

        st.success(
            f"🏆 Best Model: {best_row['Model']}"
        )

        st.header("🧠 AI Insights")

        insights = generate_ai_insights(df)

        for insight in insights:
            st.write("✅", insight)

        report_name = "AI_Data_Analysis_Report.pdf"

        generate_report(
            report_name,
            df,
            problem,
            results,
            insights
        )

        with open(report_name, "rb") as file:

            st.download_button(
                "📄 Download PDF Report",
                file,
                file_name=report_name,
                mime="application/pdf"
            )

        st.header("📊 Model Evaluation")

        metrics, matrix = evaluate_model(
            best_model,
            X_test,
            y_test,
            problem
        )

        metrics_df = pd.DataFrame(
            metrics.items(),
            columns=["Metric", "Value"]
        )

        st.dataframe(metrics_df)

        if matrix is not None:

            st.subheader("Confusion Matrix")

            st.dataframe(
                pd.DataFrame(matrix)
            )
        
        st.header("📈 Feature Importance")

        fig = plot_feature_importance(
            best_model,
            feature_names
        )

        if fig is not None:
            st.pyplot(fig)
        else:
            st.info(
                "Feature importance is not available for this model."
            )
       

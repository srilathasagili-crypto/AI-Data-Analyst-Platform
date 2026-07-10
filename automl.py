import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import accuracy_score, r2_score


def run_automl(df, target_column):

    # Separate features and target
    feature_columns = [
        col for col in df.columns
        if col != target_column
    ]

    X = df[feature_columns].copy()
    y = df[target_column].copy()


    # Handle missing values in features

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna(X[col].mode()[0])

    X = X.fillna(X.median(numeric_only=True))


    # Encode categorical features

    for col in X.select_dtypes(include=["object"]).columns:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].astype(str))


    # Encode target if categorical

    if y.dtype == "object":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y.astype(str))


    # Detect problem type

    if y.dtype == "object" or y.nunique() <= 10:
        problem = "Classification"
    else:
        problem = "Regression"


    # Split data

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    results = {}
    best_model = None
    best_score = -999


    # Classification

    if problem == "Classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier()
        }


        for name, model in models.items():

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            score = accuracy_score(
                y_test,
                pred
            )

            results[name] = score


            if score > best_score:
                best_score = score
                best_model = model


        results_df = pd.DataFrame(
            results.items(),
            columns=[
                "Model",
                "Accuracy"
            ]
        )


    # Regression

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "KNN": KNeighborsRegressor()
        }


        for name, model in models.items():

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            score = r2_score(
                y_test,
                pred
            )

            results[name] = score


            if score > best_score:
                best_score = score
                best_model = model


        results_df = pd.DataFrame(
            results.items(),
            columns=[
                "Model",
                "R2 Score"
            ]
        )


    return (
        problem,
        results_df,
        best_model,
        X_test,
        y_test,
        X.columns
    )
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pandas as pd


def evaluate_model(model, X_test, y_test, problem):

    predictions = model.predict(X_test)

    if problem == "Classification":

        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
        }

        matrix = confusion_matrix(
            y_test,
            predictions
        )

        return metrics, matrix

    else:

        metrics = {
            "MAE": mean_absolute_error(
                y_test,
                predictions
            ),
            "RMSE": mean_squared_error(
                y_test,
                predictions
            ) ** 0.5,
            "R2 Score": r2_score(
                y_test,
                predictions
            )
        }

        return metrics, None
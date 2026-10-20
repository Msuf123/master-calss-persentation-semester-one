
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def compareModels(X_train, X_test, y_train, y_test):

    models = {
        "K-NN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42)
        ),

        "ANN": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=2000,
                early_stopping=True,
                random_state=42
            )
        )
    }

    results = []
    trained_models = {}

    for name, model in models.items():

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, pos_label="Presence", zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, pos_label="Presence", zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, y_pred, pos_label="Presence", zero_division=0
            )
        })

        trained_models[name] = model

    # Rank models by F1 score, then accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["F1 Score", "Accuracy"],
        ascending=False
    ).reset_index(drop=True)

    best_name = results_df.loc[0, "Model"]
    best_model = trained_models[best_name]

    print("\nModel Performance:")
    print(results_df.round(3).to_string(index=False))

    print(f"\nBest Model: {best_name}")

    # Confusion matrix for the best model
    best_predictions = best_model.predict(X_test)

    print("\nConfusion Matrix (best model):")
    print(confusion_matrix(
        y_test,
        best_predictions,
        labels=["Absence", "Presence"]
    ))

    return results_df, best_model

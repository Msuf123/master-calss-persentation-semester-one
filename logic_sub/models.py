
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix


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

        # Train model
        model.fit(X_train, y_train)

        # Predict test data
        y_pred = model.predict(X_test)

        # Confusion matrix:
        # [[TN, FP],
        #  [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=["Absence", "Presence"]
        ).ravel()

        # Calculate metrics using mathematical formulas
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0 else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0 else 0
        )

        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        # Store results as percentages
        results.append({
            "Model": name,
            "Accuracy (%)": accuracy * 100,
            "Precision (%)": precision * 100,
            "Recall (%)": recall * 100,
            "F1 Score (%)": f1_score * 100
        })

        trained_models[name] = model

        # Print individual model results
        print(f"\n========== {name} ==========")
        print(f"True Positive (TP):  {tp}")
        print(f"True Negative (TN):  {tn}")
        print(f"False Positive (FP): {fp}")
        print(f"False Negative (FN): {fn}")
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1 Score:  {f1_score * 100:.2f}%")

    # Compare all models
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["F1 Score (%)", "Accuracy (%)"],
        ascending=False
    ).reset_index(drop=True)

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.round(2).to_string(index=False))

    # Select best model by F1 score, then accuracy
    best_name = results_df.loc[0, "Model"]
    best_model = trained_models[best_name]

    print(f"\nBest Model: {best_name}")
    print(
        f"Accuracy: "
        f"{results_df.loc[0, 'Accuracy (%)']:.2f}%"
    )

    return results_df, best_model

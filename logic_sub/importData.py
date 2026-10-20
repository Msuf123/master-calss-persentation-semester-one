import pandas as pd
from sklearn.model_selection import train_test_split


def getData():
    df = pd.read_csv("synthetic_heart_disease_270.csv")

    X = df.drop(columns=["Heart Disease"])
    y = df["Heart Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
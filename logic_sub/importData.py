
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def getData():
    df = pd.read_csv("synthetic_heart_disease_270.csv")

    X = df.drop(columns=["Heart Disease"])
    y = df["Heart Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def plot_pca(X_train, y_train):
    """Create a 2D PCA plot and return the fitted PCA and transformed data."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    labels = pd.Series(y_train).astype(str).reset_index(drop=True)

    plt.figure(figsize=(8, 6))

    for label in labels.unique():
        mask = (labels == label).to_numpy()

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=label,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3
        )

    variance = pca.explained_variance_ratio_ * 100

    plt.title("Heart Disease Dataset - 2D PCA")
    plt.xlabel(f"PC1 ({variance[0]:.1f}% variance)")
    plt.ylabel(f"PC2 ({variance[1]:.1f}% variance)")
    plt.legend(title="Heart Disease")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Important: return both objects
    return pca, X_pca


def plot_pca_two(X_train, y_train):
    """Create 2D and 3D PCA plots and return the fitted PCA and data."""

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    labels = pd.Series(y_train).astype(str).reset_index(drop=True)
    variance = pca.explained_variance_ratio_ * 100

    print(f"PC1 variance: {variance[0]:.2f}%")
    print(f"PC2 variance: {variance[1]:.2f}%")
    print(f"PC3 variance: {variance[2]:.2f}%")
    print(f"Total variance (3 PCs): {variance.sum():.2f}%")

    # 2D plot
    plt.figure(figsize=(9, 6))

    for label in labels.unique():
        mask = (labels == label).to_numpy()

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=label,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3
        )

    plt.xlabel(f"PC1 ({variance[0]:.1f}% variance)")
    plt.ylabel(f"PC2 ({variance[1]:.1f}% variance)")
    plt.title("Heart Disease Dataset - 2D PCA")
    plt.legend(title="Heart Disease")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for label in labels.unique():
        mask = (labels == label).to_numpy()

        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            X_pca[mask, 2],
            label=label,
            alpha=0.7,
            s=35,
            edgecolors="black",
            linewidths=0.3
        )

    ax.set_xlabel(f"PC1 ({variance[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}% variance)")
    ax.set_zlabel(f"PC3 ({variance[2]:.1f}% variance)")
    ax.set_title("Heart Disease Dataset - 3D PCA")
    ax.legend(title="Heart Disease")

    plt.tight_layout()
    plt.show()

    # Important: return both objects
    return pca, X_pca

def check_pca_variance(X_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Keep all principal components
    pca = PCA()
    pca.fit(X_scaled)

    variance = pca.explained_variance_ratio_ * 100
    cumulative = variance.cumsum()

    for i, (v, c) in enumerate(zip(variance, cumulative), start=1):
        print(
            f"PC{i}: {v:.2f}% variance | "
            f"Cumulative: {c:.2f}%"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(variance) + 1),
        cumulative,
        marker="o"
    )
    plt.axhline(90, linestyle="--", label="90% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance (%)")
    plt.title("PCA: Cumulative Explained Variance")
    plt.xticks(range(1, len(variance) + 1))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
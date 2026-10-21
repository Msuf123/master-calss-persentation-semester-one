from logic_sub.importData import getData
from logic_sub.models import compareModels
from logic_sub.importData import plot_pca
from  logic_sub.importData  import plot_pca_two
from logic_sub.importData import check_pca_variance
def main():
    X_train, X_test, y_train, y_test = getData()
    pca, X_pca = plot_pca(X_train, y_train)
    plot_pca_two(X_train, y_train)
    check_pca_variance(X_train)

    results, best_model = compareModels(
        X_train, X_test, y_train, y_test
    )
    print("\nTop performing model:")
    print(results.iloc[0]["Model"])


if __name__ == "__main__":
    main()
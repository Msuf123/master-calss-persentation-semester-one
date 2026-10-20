from logic_sub.importData import getData
from logic_sub.models import compareModels


def main():
    X_train, X_test, y_train, y_test = getData()

    results, best_model = compareModels(
        X_train, X_test, y_train, y_test
    )

    print("\nTop performing model:")
    print(results.iloc[0]["Model"])


if __name__ == "__main__":
    main()
from helper_function import load_data, split_data, train_and_evaluate_model
import pandas as pd


def main():
    # Step 1: Load the Iris dataset
    print("Loading dataset...")
    data = load_data()

    # Step 2: Split the dataset
    target_column = "target"
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(data, target_column)

    # Step 3: Train and evaluate multiple classifiers
    classifiers_to_train = ['logistic_regression', 'random_forest', 'svc']
    results = []

    for algorithm in classifiers_to_train:
        print(f"Training and evaluating {algorithm}...")
        result = train_and_evaluate_model(
            algorithm, X_train, y_train, X_test, y_test,
            random_state=42, max_iter=1000, kernel='linear'
        )
        metrics = result["metrics"]
        results.append({
            "Algorithm": metrics["Algorithm"],
            "Accuracy": metrics["Accuracy"]
        })

        # Print classification report and confusion matrix
        print(f"Classification Report for {algorithm}:\n", pd.DataFrame(metrics["Classification Report"]).T)
        print(f"Confusion Matrix for {algorithm}:\n", metrics["Confusion Matrix"])

    # Step 4: Display results
    print("\nSummary of Results:")
    results_df = pd.DataFrame(results)
    print(results_df)


if __name__ == "__main__":
    main()

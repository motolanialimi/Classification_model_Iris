import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Function to load data
def load_data():
    """
    Load the Iris dataset as a DataFrame.
    Returns:
        DataFrame: Features and target variables.
    """
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data


# Function to split data into train and test sets
def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    Args:
        data (DataFrame): Dataset.
        target_column (str): Target variable column name.
        test_size (float): Proportion of test data.
        random_state (int): Random seed.
    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Function to train and evaluate classifiers
def train_and_evaluate_model(algorithm, X_train, y_train, X_test, y_test, **kwargs):
    """
    Train and evaluate a classification model.
    Args:
        algorithm (str): Name of the algorithm ('logistic_regression', 'random_forest', 'svc').
        X_train (DataFrame): Training features.
        y_train (Series): Training target.
        X_test (DataFrame): Test features.
        y_test (Series): Test target.
        **kwargs: Additional parameters for the specific algorithm.
    Returns:
        dict: Dictionary containing evaluation metrics and the trained model.
    """
    if algorithm == 'logistic_regression':
        model = LogisticRegression(**kwargs)
    elif algorithm == 'random_forest':
        model = RandomForestClassifier(**kwargs)
    elif algorithm == 'svc':
        model = SVC(**kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    metrics = {
        "Algorithm": algorithm,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, output_dict=True),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

    return {"model": model, "metrics": metrics}

import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ---------------------------------------------------------
# Find the NSCLC clinical CSV
# ---------------------------------------------------------
def find_dataset():

    current_folder = os.path.dirname(
        os.path.abspath(__file__)
    )

    csv_files = glob.glob(
        os.path.join(current_folder, "*.csv")
    )

    for file in csv_files:

        if "clinical" in os.path.basename(file).lower():

            return file

    return None


# ---------------------------------------------------------
# Load and prepare the dataset
# ---------------------------------------------------------
def load_dataset():

    dataset_path = find_dataset()

    if dataset_path is None:

        raise FileNotFoundError(
            "\nClinical CSV not found.\n"
            "Place the following file in the same folder as A12.py:\n\n"
            "NSCLC-Radiomics-Lung1."
            "clinical-version3-Oct-2019(1).csv\n\n"
            "The .tcia manifest cannot be used as the "
            "ML feature dataset."
        )

    data = pd.read_csv(dataset_path)

    return data, dataset_path


# ---------------------------------------------------------
# Prepare features and target
# ---------------------------------------------------------
def prepare_data(data):

    target_column = "deadstatus.event"

    if target_column not in data.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "was not found in the dataset."
        )

    data = data.copy()

    # Remove columns that should not be used as predictors
    columns_to_remove = [
        "PatientID",
        "Survival.time"
    ]

    columns_to_remove = [
        column
        for column in columns_to_remove
        if column in data.columns
    ]

    data.drop(
        columns=columns_to_remove,
        inplace=True
    )

    # Remove rows where target is missing
    data.dropna(
        subset=[target_column],
        inplace=True
    )

    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    # Convert categorical features into numerical columns
    X = pd.get_dummies(
        X,
        drop_first=True,
        dtype=float
    )

    # Replace infinite values
    X.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Fill numerical missing values
    X = X.fillna(
        X.median(numeric_only=True)
    )

    # Handle any remaining missing values
    X = X.fillna(0)

    return X, y


# ---------------------------------------------------------
# Train MLP Classifier
# ---------------------------------------------------------
def train_mlp(X_train, y_train):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="logistic",
        solver="adam",
        learning_rate_init=0.05,
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    return model, scaler


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------
def main():

    print("=" * 70)
    print("A12 - MLPCLASSIFIER ON PROJECT DATASET")
    print("=" * 70)

    # Load dataset
    data, dataset_path = load_dataset()

    print("\nDataset:")
    print(os.path.basename(dataset_path))

    print("\nOriginal Dataset Shape:")
    print(data.shape)

    # Prepare data
    X, y = prepare_data(data)

    print("\nProcessed Feature Shape:")
    print(X.shape)

    print("\nTarget Column:")
    print("deadstatus.event")

    print("\nTarget Distribution:")
    print(y.value_counts())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining Samples:")
    print(len(X_train))

    print("\nTesting Samples:")
    print(len(X_test))

    # Train model
    model, scaler = train_mlp(
        X_train,
        y_train
    )

    # Scale testing data
    X_test_scaled = scaler.transform(
        X_test
    )

    # Predictions
    predictions = model.predict(
        X_test_scaled
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 70)
    print("MLP RESULTS")
    print("=" * 70)

    print(
        "\nNumber of Iterations:",
        model.n_iter_
    )

    print(
        "\nFinal Training Loss:",
        round(model.loss_, 6)
    )

    print(
        "\nTest Accuracy:",
        round(
            accuracy * 100,
            2
        ),
        "%"
    )

    # Confusion matrix
    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # Classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )


if __name__ == "__main__":
    main()
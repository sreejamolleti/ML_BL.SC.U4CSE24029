import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def impute_missing_values(data, method="median"):

    data = data.copy()

    for column in data.columns:

        if data[column].isnull().sum() == 0:
            continue

        if method == "mean":
            value = data[column].mean()

        elif method == "median":
            value = data[column].median()

        elif method == "mode":
            value = data[column].mode()[0]

        else:
            raise ValueError(
                "Method must be mean, median or mode"
            )

        data[column] = data[column].fillna(value)

    return data

def encode_data(data):

    data = data.copy()

    categorical_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns

    data = pd.get_dummies(
        data,
        columns=categorical_columns,
        drop_first=True
    )

    return data

file_path = r"C:\Users\mahik\Downloads\ML lab\NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"

data = pd.read_csv(file_path)

data = data.drop(
    "PatientID",
    axis=1
)

data = data.drop(
    "Survival.time",
    axis=1
)

X = data.drop(
    "deadstatus.event",
    axis=1
)

y = data["deadstatus.event"]

X = encode_data(X)

X = impute_missing_values(
    X,
    method="median"
)

X = X.astype(float)

y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

knn_model = KNeighborsClassifier(
    n_neighbors=3
)

knn_model.fit(
    X_train,
    y_train
)

print("kNN classifier trained successfully.")

print("\nNumber of neighbors (k):",
      knn_model.n_neighbors)

print("Training samples:",
      X_train.shape[0])

print("Testing samples:",
      X_test.shape[0])
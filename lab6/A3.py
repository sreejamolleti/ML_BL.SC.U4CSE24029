import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from A1 import knn_predict


file_path = r"C:\Users\mahik\Downloads\ML lab\NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"

data = pd.read_csv(file_path)

data = data.drop(
    ["PatientID", "Survival.time"],
    axis=1
)

X = data.drop(
    "deadstatus.event",
    axis=1
)

y = data["deadstatus.event"]


categorical = X.select_dtypes(
    include=["object", "string", "category"]
).columns

X = pd.get_dummies(
    X,
    columns=categorical,
    drop_first=True
)


for column in X.columns:

    if X[column].isnull().sum() > 0:
        X[column] = X[column].fillna(
            X[column].median()
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


X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


k = 3
runs = 10


def my_knn():

    return knn_predict(
        X_train,
        y_train,
        X_test,
        k=k
    )


def sklearn_knn():

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train,
        y_train
    )

    return model.predict(
        X_test
    )


def genai_knn():

    return knn_predict(
        X_train,
        y_train,
        X_test,
        k=k
    )


def evaluate_model(name, function):

    predictions = function()

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    times = []

    for i in range(runs):

        start = time.perf_counter()

        function()

        end = time.perf_counter()

        times.append(
            end - start
        )

    average_time = np.mean(times)

    return [
        name,
        accuracy,
        precision,
        recall,
        f1,
        average_time
    ]


results = []


results.append(
    evaluate_model(
        "My kNN",
        my_knn
    )
)


results.append(
    evaluate_model(
        "Scikit-learn kNN",
        sklearn_knn
    )
)


results.append(
    evaluate_model(
        "GenAI kNN",
        genai_knn
    )
)


result_table = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Average Time (seconds)"
    ]
)


print("\nPerformance Comparison")
print(
    result_table.to_string(
        index=False
    )
)
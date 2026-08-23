import numpy as np
import pandas as pd
from collections import Counter


def encode_data(data):
    categorical = data.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    return pd.get_dummies(
        data,
        columns=categorical,
        drop_first=True
    )


def impute_data(data):
    data = data.copy()

    for column in data.columns:
        if data[column].isnull().sum() > 0:
            data[column] = data[column].fillna(
                data[column].median()
            )

    return data


def calculate_distance(a, b, metric="euclidean"):

    if metric == "euclidean":
        return np.sqrt(np.sum((a - b) ** 2))

    elif metric == "manhattan":
        return np.sum(np.abs(a - b))

    else:
        raise ValueError("Invalid distance metric")


def bubble_sort(values):
    values = values.copy()

    for i in range(len(values)):
        for j in range(len(values) - i - 1):

            if values[j][0] > values[j + 1][0]:
                values[j], values[j + 1] = (
                    values[j + 1],
                    values[j]
                )

    return values


def selection_sort(values):
    values = values.copy()

    for i in range(len(values)):

        minimum = i

        for j in range(i + 1, len(values)):

            if values[j][0] < values[minimum][0]:
                minimum = j

        values[i], values[minimum] = (
            values[minimum],
            values[i]
        )

    return values


def insertion_sort(values):
    values = values.copy()

    for i in range(1, len(values)):

        current = values[i]
        j = i - 1

        while j >= 0 and values[j][0] > current[0]:
            values[j + 1] = values[j]
            j -= 1

        values[j + 1] = current

    return values


def sort_distances(values, method="bubble"):

    if method == "bubble":
        return bubble_sort(values)

    elif method == "selection":
        return selection_sort(values)

    elif method == "insertion":
        return insertion_sort(values)

    else:
        raise ValueError("Invalid sorting method")


def get_neighbors(
        X_train,
        y_train,
        point,
        k=3,
        metric="euclidean",
        sort_method="bubble"
    ):

    distances = []

    for i in range(len(X_train)):

        d = calculate_distance(
            X_train[i],
            point,
            metric
        )

        distances.append(
            (d, y_train[i])
        )

    distances = sort_distances(
        distances,
        sort_method
    )

    return distances[:k]


def predict_class(neighbors):

    votes = Counter()

    for distance, label in neighbors:
        votes[label] += 1

    return votes.most_common(1)[0][0]


def knn_predict(
        X_train,
        y_train,
        X_test,
        k=3,
        metric="euclidean",
        sort_method="bubble"
    ):

    predictions = []

    for point in X_test:

        neighbors = get_neighbors(
            X_train,
            y_train,
            point,
            k,
            metric,
            sort_method
        )

        predictions.append(
            predict_class(neighbors)
        )

    return np.array(predictions)


def weighted_predict_class(neighbors):

    weights = {}

    for distance, label in neighbors:

        if distance == 0:
            return label

        weight = 1 / distance

        if label not in weights:
            weights[label] = 0

        weights[label] += weight

    return max(
        weights,
        key=weights.get
    )


def weighted_knn_predict(
        X_train,
        y_train,
        X_test,
        k=3
    ):

    predictions = []

    for point in X_test:

        neighbors = get_neighbors(
            X_train,
            y_train,
            point,
            k
        )

        predictions.append(
            weighted_predict_class(neighbors)
        )

    return np.array(predictions)


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

X = encode_data(X)
X = impute_data(X)

X = X.astype(float).to_numpy()
y = y.astype(int).to_numpy()

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

prediction = knn_predict(
    X[1:],
    y[1:],
    X[:1],
    k=3
)

print("Prediction for first sample:", prediction[0])
print("Actual class:", y[0])
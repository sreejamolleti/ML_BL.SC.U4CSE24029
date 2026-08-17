import numpy as np
import pandas as pd
from collections import Counter

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


def calculate_distance(point1, point2, metric="euclidean"):

    if metric == "euclidean":

        distance = np.sqrt(
            np.sum((point1 - point2) ** 2)
        )

    elif metric == "manhattan":

        distance = np.sum(
            np.abs(point1 - point2)
        )

    else:

        raise ValueError(
            "Metric must be euclidean or manhattan"
        )

    return distance


def bubble_sort(distance_list):

    arr = distance_list.copy()
    n = len(arr)

    for i in range(n):

        for j in range(0, n - i - 1):

            if arr[j][0] > arr[j + 1][0]:

                arr[j], arr[j + 1] = (
                    arr[j + 1],
                    arr[j]
                )

    return arr

def selection_sort(distance_list): 
    arr = distance_list.copy()
    n = len(arr)

    for i in range(n):

        minimum = i

        for j in range(i + 1, n):

            if arr[j][0] < arr[minimum][0]:

                minimum = j

        arr[i], arr[minimum] = (
            arr[minimum],
            arr[i]
        )

    return arr


def insertion_sort(distance_list):
    arr = distance_list.copy()

    for i in range(1, len(arr)):

        current = arr[i]
        j = i - 1

        while j >= 0 and arr[j][0] > current[0]:

            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = current

    return arr

def sort_distances(distance_list, sorting_algorithm="bubble"):

    if sorting_algorithm == "bubble":

        return bubble_sort(distance_list)

    elif sorting_algorithm == "selection":

        return selection_sort(distance_list)

    elif sorting_algorithm == "insertion":

        return insertion_sort(distance_list)

    else:

        raise ValueError(
            "Choose bubble, selection or insertion"
        )


def find_neighbors(
        X_train,
        y_train,
        test_point,
        k=3,
        metric="euclidean",
        sorting_algorithm="bubble"
    ):

    distance_list = []

    for i in range(len(X_train)):

        distance = calculate_distance(
            X_train[i],
            test_point,
            metric
        )

        distance_list.append(
            (distance, y_train[i], i)
        )

    sorted_distances = sort_distances(
        distance_list,
        sorting_algorithm
    )
    neighbors = sorted_distances[:k]

    return neighbors

def predict_class(neighbors):

    class_counts = Counter()

    for distance, label, index in neighbors:

        class_counts[label] += 1

    maximum_votes = max(
        class_counts.values()
    )

    candidate_classes = [
        label
        for label, count in class_counts.items()
        if count == maximum_votes
    ]

    if len(candidate_classes) == 1:

        return candidate_classes[0]

    distance_sum = {}

    for label in candidate_classes:

        distance_sum[label] = sum(
            distance
            for distance, neighbour_label, index
            in neighbors
            if neighbour_label == label
        )

    return min(
        distance_sum,
        key=distance_sum.get
    )

def knn_predict(
        X_train,
        y_train,
        test_point,
        k=3,
        metric="euclidean",
        sorting_algorithm="bubble"
    ):

    neighbors = find_neighbors(
        X_train,
        y_train,
        test_point,
        k,
        metric,
        sorting_algorithm
    )

    prediction = predict_class(
        neighbors
    )

    return prediction


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

y = data["deadstatus.event"]

X = data.drop(
    "deadstatus.event",
    axis=1
)

y = y.astype(int).values

X = encode_data(X)

X = impute_missing_values(
    X,
    method="median"
)

X = X.astype(float).values


test_point = X[0]

X_train = X[1:]
y_train = y[1:]

prediction = knn_predict(
    X_train,
    y_train,
    test_point,
    k=3,
    metric="euclidean",
    sorting_algorithm="bubble"
)

print("Predicted class:", prediction)
print("Actual class:", y[0])

neighbors = find_neighbors(
    X_train,
    y_train,
    test_point,
    k=3,
    metric="euclidean",
    sorting_algorithm="bubble"
)

print("\nNearest Neighbors:")

for distance, label, index in neighbors:

    print(
        "Distance:",
        round(distance, 4),
        "Class:",
        label
    )
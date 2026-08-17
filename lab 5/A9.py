import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
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
        include=["object", "string", "category"]
    ).columns

    data = pd.get_dummies(
        data,
        columns=categorical_columns,
        drop_first=True
    )

    return data


def calculate_distance(point1, point2, metric="euclidean"):

    if metric == "euclidean":
        return np.sqrt(
            np.sum((point1 - point2) ** 2)
        )

    elif metric == "manhattan":
        return np.sum(
            np.abs(point1 - point2)
        )

    else:
        raise ValueError(
            "Metric must be euclidean or manhattan"
        )


def bubble_sort(distance_list):
    arr = distance_list.copy()

    for i in range(len(arr)):

        for j in range(0, len(arr) - i - 1):

            if arr[j][0] > arr[j + 1][0]:

                arr[j], arr[j + 1] = (
                    arr[j + 1],
                    arr[j]
                )

    return arr


def selection_sort(distance_list):
    arr = distance_list.copy()

    for i in range(len(arr)):

        minimum = i

        for j in range(i + 1, len(arr)):

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


def sort_distances(
        distance_list,
        sorting_algorithm="bubble"
    ):

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

    return sorted_distances[:k]


def fit(X, y):

    return {
        "X_train": np.asarray(X),
        "y_train": np.asarray(y)
    }


def predict(
        model,
        X,
        k=3,
        metric="euclidean",
        sorting_algorithm="bubble"
    ):

    predictions = []

    X_train = model["X_train"]
    y_train = model["y_train"]

    for test_point in X:

        neighbors = find_neighbors(
            X_train,
            y_train,
            test_point,
            k,
            metric,
            sorting_algorithm
        )

        predictions.append(
            predict_class(neighbors)
        )

    return np.array(predictions)


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


def score(
        model,
        X,
        y,
        k=3,
        metric="euclidean",
        sorting_algorithm="bubble"
    ):

    predictions = predict(
        model,
        X,
        k,
        metric,
        sorting_algorithm
    )

    return np.mean(
        predictions == np.asarray(y)
    )


def weighted_predict_class(neighbors):

    class_weights = {}

    for distance, label, index in neighbors:

        if distance == 0:
            return label

        weight = 1 / distance

        if label not in class_weights:
            class_weights[label] = 0

        class_weights[label] += weight

    maximum_weight = max(
        class_weights.values()
    )

    candidate_classes = [
        label
        for label, weight in class_weights.items()
        if weight == maximum_weight
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


def weighted_predict(
        model,
        X,
        k=3,
        metric="euclidean",
        sorting_algorithm="bubble"
    ):

    predictions = []

    X_train = model["X_train"]
    y_train = model["y_train"]

    for test_point in X:

        neighbors = find_neighbors(
            X_train,
            y_train,
            test_point,
            k,
            metric,
            sorting_algorithm
        )

        predictions.append(
            weighted_predict_class(neighbors)
        )

    return np.array(predictions)


def weighted_score(
        model,
        X,
        y,
        k=3,
        metric="euclidean",
        sorting_algorithm="bubble"
    ):

    predictions = weighted_predict(
        model,
        X,
        k,
        metric,
        sorting_algorithm
    )

    return np.mean(
        predictions == np.asarray(y)
    )


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

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

model = fit(
    X_train,
    y_train
)

k_values = range(1, 16, 2)

normal_accuracies = []
weighted_accuracies = []
sklearn_accuracies = []


for k in k_values:

    normal_accuracy = score(
        model,
        X_test,
        y_test,
        k=k,
        metric="euclidean",
        sorting_algorithm="bubble"
    )

    normal_accuracies.append(
        normal_accuracy
    )

    weighted_accuracy = weighted_score(
        model,
        X_test,
        y_test,
        k=k,
        metric="euclidean",
        sorting_algorithm="bubble"
    )

    weighted_accuracies.append(
        weighted_accuracy
    )

    sklearn_model = KNeighborsClassifier(
        n_neighbors=k
    )

    sklearn_model.fit(
        X_train,
        y_train
    )

    sklearn_accuracy = sklearn_model.score(
        X_test,
        y_test
    )

    sklearn_accuracies.append(
        sklearn_accuracy
    )


results = pd.DataFrame({
    "k": list(k_values),
    "Normal kNN": normal_accuracies,
    "Weighted kNN": weighted_accuracies,
    "Scikit-learn kNN": sklearn_accuracies
})

print("A9 Accuracy Comparison:")
print(results)

print("\nWeighted kNN Accuracies:")
print(weighted_accuracies)


plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    normal_accuracies,
    marker="o",
    label="Normal kNN"
)

plt.plot(
    k_values,
    weighted_accuracies,
    marker="s",
    label="Weighted kNN"
)

plt.plot(
    k_values,
    sklearn_accuracies,
    marker="^",
    label="Scikit-learn kNN"
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("Comparison of Normal and Weighted kNN")

plt.xticks(list(k_values))
plt.legend()
plt.grid(True)

plt.show()
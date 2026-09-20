# ============================================================
# A5 - OWN DECISION TREE MODULE
# Entropy + Information Gain
# Dataset: NSCLC-Radiomics Lung1 Clinical Dataset
# ============================================================

import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# DATASET
# ============================================================

CLINICAL_FILE = "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"

TARGET = "deadstatus.event"


# ============================================================
# TREE NODE
# ============================================================

class TreeNode:

    def __init__(self, feature=None, prediction=None):

        self.feature = feature

        self.prediction = prediction

        self.children = {}


# ============================================================
# CUSTOM DECISION TREE
# ============================================================

class MyDecisionTree:

    def __init__(
        self,
        max_depth=4,
        min_samples_split=5
    ):

        self.max_depth = max_depth

        self.min_samples_split = min_samples_split

        self.root = None

        self.features = []


    # ========================================================
    # ENTROPY
    # ========================================================

    def entropy(self, y):

        y = pd.Series(y).dropna()

        probabilities = y.value_counts(
            normalize=True
        )

        entropy_value = 0.0

        for probability in probabilities:

            if probability > 0:

                entropy_value -= (
                    probability *
                    math.log2(probability)
                )

        return entropy_value


    # ========================================================
    # INFORMATION GAIN
    # ========================================================

    def information_gain(
        self,
        X,
        y,
        feature
    ):

        parent_entropy = self.entropy(y)

        weighted_entropy = 0.0

        values = X[feature].unique()

        for value in values:

            mask = (
                X[feature] == value
            )

            subset_y = y[mask]

            if len(subset_y) == 0:

                continue

            weight = (
                len(subset_y) /
                len(y)
            )

            subset_entropy = self.entropy(
                subset_y
            )

            weighted_entropy += (
                weight *
                subset_entropy
            )

        gain = (
            parent_entropy -
            weighted_entropy
        )

        return gain


    # ========================================================
    # FIND BEST FEATURE
    # ========================================================

    def find_best_feature(
        self,
        X,
        y,
        available_features
    ):

        best_feature = None

        best_gain = -1

        for feature in available_features:

            gain = self.information_gain(
                X,
                y,
                feature
            )

            if gain > best_gain:

                best_gain = gain

                best_feature = feature

        return best_feature, best_gain


    # ========================================================
    # BUILD TREE
    # ========================================================

    def build_tree(
        self,
        X,
        y,
        available_features,
        depth=0
    ):

        # ----------------------------------------------------
        # Condition 1:
        # All samples belong to one class
        # ----------------------------------------------------

        if len(y.unique()) == 1:

            return TreeNode(
                prediction=y.iloc[0]
            )


        # ----------------------------------------------------
        # Condition 2:
        # No features remaining
        # ----------------------------------------------------

        if len(available_features) == 0:

            return TreeNode(
                prediction=y.mode()[0]
            )


        # ----------------------------------------------------
        # Condition 3:
        # Maximum depth reached
        # ----------------------------------------------------

        if depth >= self.max_depth:

            return TreeNode(
                prediction=y.mode()[0]
            )


        # ----------------------------------------------------
        # Condition 4:
        # Too few samples
        # ----------------------------------------------------

        if len(y) < self.min_samples_split:

            return TreeNode(
                prediction=y.mode()[0]
            )


        # ----------------------------------------------------
        # Find best feature
        # ----------------------------------------------------

        best_feature, best_gain = (
            self.find_best_feature(
                X,
                y,
                available_features
            )
        )


        # ----------------------------------------------------
        # No useful split
        # ----------------------------------------------------

        if (
            best_feature is None
            or best_gain <= 0
        ):

            return TreeNode(
                prediction=y.mode()[0]
            )


        # ----------------------------------------------------
        # Create decision node
        # ----------------------------------------------------

        node = TreeNode(
            feature=best_feature
        )


        remaining_features = [

            feature

            for feature in available_features

            if feature != best_feature

        ]


        # ----------------------------------------------------
        # Create child nodes
        # ----------------------------------------------------

        for value in X[best_feature].unique():

            mask = (
                X[best_feature] == value
            )

            X_subset = X.loc[mask]

            y_subset = y.loc[mask]


            if len(y_subset) == 0:

                continue


            child = self.build_tree(

                X_subset,

                y_subset,

                remaining_features,

                depth + 1

            )


            node.children[value] = child


        return node


    # ========================================================
    # FIT
    # ========================================================

    def fit(self, X, y):

        self.features = list(
            X.columns
        )

        self.root = self.build_tree(

            X,

            y,

            self.features,

            depth=0

        )

        return self


    # ========================================================
    # PREDICT ONE SAMPLE
    # ========================================================

    def predict_one(
        self,
        row,
        node=None
    ):

        if node is None:

            node = self.root


        # ----------------------------------------------------
        # Leaf node
        # ----------------------------------------------------

        if node.prediction is not None:

            return node.prediction


        # ----------------------------------------------------
        # Get feature value
        # ----------------------------------------------------

        value = row[node.feature]


        # ----------------------------------------------------
        # Unknown value
        # ----------------------------------------------------

        if value not in node.children:

            # Default prediction
            # when unseen category occurs

            return 0


        # ----------------------------------------------------
        # Continue down tree
        # ----------------------------------------------------

        return self.predict_one(

            row,

            node.children[value]

        )


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, X):

        predictions = []

        for _, row in X.iterrows():

            prediction = self.predict_one(
                row
            )

            predictions.append(
                prediction
            )

        return np.array(
            predictions
        )


    # ========================================================
    # PRINT TREE
    # ========================================================

    def print_tree(
        self,
        node=None,
        level=0
    ):

        if node is None:

            node = self.root


        indentation = (
            "    " * level
        )


        # ----------------------------------------------------
        # Leaf
        # ----------------------------------------------------

        if node.prediction is not None:

            print(
                indentation +
                "Predict ->",
                node.prediction
            )

            return


        # ----------------------------------------------------
        # Decision node
        # ----------------------------------------------------

        print(
            indentation +
            "Feature ->",
            node.feature
        )


        # ----------------------------------------------------
        # Branches
        # ----------------------------------------------------

        for value, child in node.children.items():

            print(
                indentation +
                "  Value ->",
                value
            )

            self.print_tree(
                child,
                level + 1
            )


# ============================================================
# LOAD DATASET
# ============================================================

print("==========================================")
print("A5 - CUSTOM DECISION TREE")
print("==========================================")


try:

    df = pd.read_csv(
        CLINICAL_FILE
    )

except FileNotFoundError:

    print(
        "\nERROR: Dataset not found."
    )

    print(
        "Expected file:"
    )

    print(
        CLINICAL_FILE
    )

    exit()


print(
    "\nDataset Shape:",
    df.shape
)


# ============================================================
# SELECT FEATURES
# ============================================================

# PatientID is an identifier and is excluded.
#
# Survival.time is excluded from prediction because
# it is outcome-related and can cause data leakage.

features = [

    "age",

    "clinical.T.Stage",

    "Clinical.N.Stage",

    "Clinical.M.Stage",

    "Overall.Stage",

    "Histology",

    "gender"

]


X = df[
    features
].copy()


y = df[
    TARGET
].copy()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

for column in X.columns:

    if pd.api.types.is_numeric_dtype(
        X[column]
    ):

        X[column] = X[column].fillna(
            X[column].median()
        )

    else:

        X[column] = X[column].fillna(
            "Missing"
        )


# ============================================================
# BIN NUMERICAL FEATURES
# ============================================================

numerical_features = [

    "age",

    "clinical.T.Stage",

    "Clinical.N.Stage",

    "Clinical.M.Stage"

]


for feature in numerical_features:

    X[feature] = pd.cut(

        X[feature],

        bins=4,

        labels=[
            "Bin_1",
            "Bin_2",
            "Bin_3",
            "Bin_4"
        ],

        include_lowest=True

    ).astype(str)


# ============================================================
# CONVERT ALL FEATURES TO STRING
# ============================================================

X = X.astype(str)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "\nTraining Samples:",
    len(X_train)
)

print(
    "Testing Samples:",
    len(X_test)
)


# ============================================================
# CREATE CUSTOM DECISION TREE
# ============================================================

tree = MyDecisionTree(

    max_depth=4,

    min_samples_split=5

)


# ============================================================
# TRAIN
# ============================================================

tree.fit(

    X_train,

    y_train

)


# ============================================================
# PREDICTION
# ============================================================

y_pred = tree.predict(
    X_test
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")


print(
    "\nAccuracy:",
    accuracy
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# PRINT DECISION TREE
# ============================================================

print("\n==========================================")
print("DECISION TREE STRUCTURE")
print("==========================================")


tree.print_tree()


# ============================================================
# ROOT NODE
# ============================================================

print("\n==========================================")
print("ROOT NODE")
print("==========================================")


if tree.root is not None:

    if tree.root.feature is not None:

        print(
            "Root Feature:",
            tree.root.feature
        )

    else:

        print(
            "Root is a leaf node."
        )


# ============================================================
# END
# ============================================================

print("\n==========================================")
print("A5 COMPLETED SUCCESSFULLY")
print("==========================================")
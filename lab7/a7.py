# ============================================================
# A7 - DECISION BOUNDARY USING TWO FEATURES
# Dataset: NSCLC-Radiomics Lung1 Clinical Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# DATASET FILE
# ============================================================

CLINICAL_FILE = "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"

TARGET = "deadstatus.event"


# ============================================================
# SELECT EXACTLY TWO FEATURES
# ============================================================

FEATURE_1 = "age"
FEATURE_2 = "clinical.T.Stage"


# ============================================================
# LOAD DATASET
# ============================================================

print("==========================================")
print("A7 - DECISION BOUNDARY")
print("==========================================")


try:

    df = pd.read_csv(
        CLINICAL_FILE
    )

except FileNotFoundError:

    print(
        "\nERROR: Dataset file not found."
    )

    print(
        "Expected file:",
        CLINICAL_FILE
    )

    exit()


print(
    "\nDataset Shape:",
    df.shape
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [

    FEATURE_1,
    FEATURE_2,
    TARGET

]


for column in required_columns:

    if column not in df.columns:

        print(
            f"\nERROR: Column '{column}' not found."
        )

        print(
            "\nAvailable columns:"
        )

        print(
            df.columns.tolist()
        )

        exit()


# ============================================================
# SELECT DATA
# ============================================================

data = df[
    [
        FEATURE_1,
        FEATURE_2,
        TARGET
    ]
].copy()


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

data[FEATURE_1] = pd.to_numeric(
    data[FEATURE_1],
    errors="coerce"
)


data[FEATURE_2] = pd.to_numeric(
    data[FEATURE_2],
    errors="coerce"
)


data[TARGET] = pd.to_numeric(
    data[TARGET],
    errors="coerce"
)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

data = data.dropna()


print(
    "\nSamples after removing missing values:",
    len(data)
)


# ============================================================
# CREATE X AND Y
# ============================================================

X = data[
    [
        FEATURE_1,
        FEATURE_2
    ]
]


y = data[
    TARGET
].astype(int)


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

print(
    "\nTarget Distribution:"
)

print(
    y.value_counts()
)


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
# CREATE DECISION TREE
# ============================================================

dt = DecisionTreeClassifier(

    criterion="entropy",

    max_depth=4,

    min_samples_split=5,

    min_samples_leaf=2,

    random_state=42

)


# ============================================================
# TRAIN DECISION TREE
# ============================================================

dt.fit(

    X_train,

    y_train

)


print(
    "\nDecision Tree trained successfully."
)


# ============================================================
# TEST PREDICTION
# ============================================================

y_pred = dt.predict(
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


print(
    "Accuracy Percentage:",
    round(
        accuracy * 100,
        2
    ),
    "%"
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
# CREATE MESH GRID
# ============================================================

x_min = X[FEATURE_1].min() - 2
x_max = X[FEATURE_1].max() + 2


y_min = X[FEATURE_2].min() - 1
y_max = X[FEATURE_2].max() + 1


xx, yy = np.meshgrid(

    np.linspace(
        x_min,
        x_max,
        500
    ),

    np.linspace(
        y_min,
        y_max,
        500
    )

)


# ============================================================
# PREDICT GRID POINTS
# ============================================================

grid_points = np.c_[

    xx.ravel(),

    yy.ravel()

]


Z = dt.predict(
    grid_points
)


Z = Z.reshape(
    xx.shape
)


# ============================================================
# PLOT DECISION BOUNDARY
# ============================================================

plt.figure(
    figsize=(11, 8)
)


# Decision regions

plt.contourf(

    xx,

    yy,

    Z,

    alpha=0.25

)


# ============================================================
# PLOT DATA POINTS
# ============================================================

classes = sorted(
    y.unique()
)


for class_value in classes:

    mask = (
        y == class_value
    )


    plt.scatter(

        X.loc[mask, FEATURE_1],

        X.loc[mask, FEATURE_2],

        label=f"Class {class_value}",

        edgecolor="black",

        s=45

    )


# ============================================================
# LABELS
# ============================================================

plt.xlabel(
    "Age",
    fontsize=12
)


plt.ylabel(
    "Clinical T Stage",
    fontsize=12
)


plt.title(
    "Decision Boundary of Decision Tree\n"
    "Using Age and Clinical T Stage",
    fontsize=15
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


# ============================================================
# SAVE GRAPH
# ============================================================

OUTPUT_FILE = "A7_decision_boundary.png"


plt.savefig(

    OUTPUT_FILE,

    dpi=300,

    bbox_inches="tight"

)


print("\n==========================================")
print("DECISION BOUNDARY")
print("==========================================")


print(
    "\nFeatures used:"
)

print(
    "1.",
    FEATURE_1
)

print(
    "2.",
    FEATURE_2
)


print(
    "\nDecision boundary saved as:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# DISPLAY GRAPH
# ============================================================

plt.show()


# ============================================================
# TREE INFORMATION
# ============================================================

print("\n==========================================")
print("TREE INFORMATION")
print("==========================================")


print(
    "Tree Depth:",
    dt.get_depth()
)


print(
    "Number of Leaves:",
    dt.get_n_leaves()
)


# ============================================================
# COMPLETION
# ============================================================

print("\n==========================================")
print("A7 COMPLETED SUCCESSFULLY")
print("==========================================")
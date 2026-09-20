# ============================================================
# A6 - DECISION TREE VISUALIZATION
# Dataset: NSCLC-Radiomics Lung1 Clinical Dataset
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

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
# LOAD DATASET
# ============================================================

print("==========================================")
print("A6 - DECISION TREE VISUALIZATION")
print("==========================================")

try:

    df = pd.read_csv(
        CLINICAL_FILE
    )

except FileNotFoundError:

    print("\nERROR: Dataset file not found.")

    print(
        "Expected file:",
        CLINICAL_FILE
    )

    exit()


print(
    "\nDataset Shape:",
    df.shape
)


print(
    "\nDataset Columns:"
)

print(
    df.columns.tolist()
)


# ============================================================
# SELECT FEATURES
# ============================================================

# PatientID is an identifier and is excluded.
#
# Survival.time is excluded because it is related to the
# outcome and can cause data leakage.

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
# ONE-HOT ENCODING
# ============================================================

X = pd.get_dummies(
    X,
    drop_first=False
)


# ============================================================
# CONVERT DATA TO NUMERIC
# ============================================================

X = X.astype(float)


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
# TRAIN MODEL
# ============================================================

dt.fit(

    X_train,

    y_train

)


print(
    "\nDecision Tree trained successfully."
)


# ============================================================
# PREDICTION
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
# DECISION TREE INFORMATION
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


print(
    "Number of Nodes:",
    dt.tree_.node_count
)


# ============================================================
# VISUALIZE DECISION TREE
# ============================================================

print("\nGenerating Decision Tree visualization...")


plt.figure(
    figsize=(28, 18)
)


plot_tree(

    dt,

    feature_names=X.columns,

    class_names=[
        "Alive",
        "Dead"
    ],

    filled=True,

    rounded=True,

    proportion=False,

    precision=3,

    fontsize=8

)


plt.title(
    "NSCLC-Radiomics Decision Tree",
    fontsize=18
)


plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

OUTPUT_FILE = "A6_decision_tree.png"


plt.savefig(

    OUTPUT_FILE,

    dpi=300,

    bbox_inches="tight"

)


print(
    "\nDecision Tree saved as:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# DISPLAY FIGURE
# ============================================================

plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        dt.feature_importances_

})


importance = importance.sort_values(

    by="Importance",

    ascending=False

)


print("\n==========================================")
print("FEATURE IMPORTANCE")
print("==========================================")


print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# COMPLETION
# ============================================================

print("\n==========================================")
print("A6 COMPLETED SUCCESSFULLY")
print("==========================================")
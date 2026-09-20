# ============================================================
# A8 - HYPERPARAMETER TUNING USING GridSearchCV
# Dataset: NSCLC-Radiomics Lung1 Clinical Dataset
# ============================================================

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# DATASET
# ============================================================

CLINICAL_FILE = "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"

TARGET = "deadstatus.event"


# ============================================================
# LOAD DATASET
# ============================================================

print("==========================================")
print("A8 - HYPERPARAMETER TUNING")
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


# ============================================================
# SELECT FEATURES
# ============================================================

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
# CONVERT TO NUMERIC
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
# BASE DECISION TREE
# ============================================================

dt = DecisionTreeClassifier(
    random_state=42
)


# ============================================================
# REDUCED HYPERPARAMETER GRID
# ============================================================

# The previous grid had:
#
# 3 criteria
# × 8 max_depth values
# × 4 min_samples_split values
# × 4 min_samples_leaf values
# × 3 max_features values
#
# = 1152 combinations
#
# This reduced grid gives a practical search.

param_grid = {

    "criterion": [
        "gini",
        "entropy"
    ],

    "max_depth": [
        2,
        3,
        4,
        5,
        6,
        None
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ]

}


# ============================================================
# COUNT COMBINATIONS
# ============================================================

number_of_combinations = (

    len(param_grid["criterion"])

    * len(param_grid["max_depth"])

    * len(param_grid["min_samples_split"])

    * len(param_grid["min_samples_leaf"])

)


print("\n==========================================")
print("GRID SEARCH CONFIGURATION")
print("==========================================")


print(
    "Number of parameter combinations:",
    number_of_combinations
)


print(
    "Cross-validation folds:",
    5
)


print(
    "Total model fits:",
    number_of_combinations * 5
)


# ============================================================
# GRID SEARCH
# ============================================================

grid_search = GridSearchCV(

    estimator=dt,

    param_grid=param_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,

    verbose=1

)


# ============================================================
# RUN GRID SEARCH
# ============================================================

print("\n==========================================")
print("STARTING GRID SEARCH")
print("==========================================")


grid_search.fit(

    X_train,

    y_train

)


print(
    "\nGrid Search completed successfully."
)


# ============================================================
# BEST PARAMETERS
# ============================================================

print("\n==========================================")
print("BEST HYPERPARAMETERS")
print("==========================================")


print(
    "\nBest Parameters:"
)


for parameter, value in (
    grid_search.best_params_.items()
):

    print(
        f"{parameter}: {value}"
    )


# ============================================================
# BEST CV SCORE
# ============================================================

best_cv_score = (
    grid_search.best_score_
)


print(
    "\nBest Cross-Validation Accuracy:",
    best_cv_score
)


print(
    "Best CV Accuracy Percentage:",
    round(
        best_cv_score * 100,
        2
    ),
    "%"
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = (
    grid_search.best_estimator_
)


# ============================================================
# TEST SET PREDICTION
# ============================================================

y_pred = best_model.predict(

    X_test

)


# ============================================================
# TEST ACCURACY
# ============================================================

test_accuracy = accuracy_score(

    y_test,

    y_pred

)


print("\n==========================================")
print("TEST SET PERFORMANCE")
print("==========================================")


print(
    "\nTest Accuracy:",
    test_accuracy
)


print(
    "Test Accuracy Percentage:",
    round(
        test_accuracy * 100,
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
# GRID SEARCH RESULTS
# ============================================================

results = pd.DataFrame(

    grid_search.cv_results_

)


results = results.sort_values(

    by="mean_test_score",

    ascending=False

)


# ============================================================
# TOP 10 RESULTS
# ============================================================

print("\n==========================================")
print("TOP 10 PARAMETER COMBINATIONS")
print("==========================================")


top_results = results[

    [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "params"
    ]

].head(10)


print(

    top_results.to_string(
        index=False
    )

)


# ============================================================
# SAVE GRID SEARCH RESULTS
# ============================================================

results_file = (
    "A8_gridsearch_results.csv"
)


results.to_csv(

    results_file,

    index=False

)


print(
    "\nComplete GridSearchCV results saved as:"
)

print(
    results_file
)


# ============================================================
# BEST MODEL INFORMATION
# ============================================================

print("\n==========================================")
print("BEST MODEL INFORMATION")
print("==========================================")


print(
    "Tree Depth:",
    best_model.get_depth()
)


print(
    "Number of Leaves:",
    best_model.get_n_leaves()
)


print(
    "Number of Nodes:",
    best_model.tree_.node_count
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature":
        X.columns,

    "Importance":
        best_model.feature_importances_

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
# SAVE FEATURE IMPORTANCE
# ============================================================

importance.to_csv(

    "A8_feature_importance.csv",

    index=False

)


# ============================================================
# COMPLETION
# ============================================================

print("\n==========================================")
print("A8 COMPLETED SUCCESSFULLY")
print("==========================================")
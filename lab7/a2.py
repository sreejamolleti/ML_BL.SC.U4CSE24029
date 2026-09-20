# ============================================================
# A2 - GINI INDEX
# Dataset: NSCLC-Radiomics Lung1 Clinical Dataset
# ============================================================

import pandas as pd


CLINICAL_FILE = "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

df = pd.read_csv(CLINICAL_FILE)


# ------------------------------------------------------------
# Gini Index Function
# ------------------------------------------------------------

def calculate_gini(values):
    """
    Calculates Gini Index.

    Gini = 1 - Sum(pj^2)
    """

    values = pd.Series(values).dropna()

    probabilities = values.value_counts(normalize=True)

    gini = 1.0 - sum(probability ** 2
                      for probability in probabilities)

    return gini


# ------------------------------------------------------------
# Calculate Gini for Target
# ------------------------------------------------------------

target = df["deadstatus.event"]

gini_value = calculate_gini(target)


# ------------------------------------------------------------
# Display result
# ------------------------------------------------------------

print("==========================================")
print("A2 - GINI INDEX")
print("==========================================")

print("Target Variable: deadstatus.event")

print("\nClass Distribution:")
print(target.value_counts())

print("\nClass Probabilities:")
print(target.value_counts(normalize=True))

print("\nGini Index =", gini_value)


# ------------------------------------------------------------
# Gini for individual categorical/binned attributes
# ------------------------------------------------------------

def equal_width_binning(series, n_bins=4):

    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=n_bins,
        include_lowest=True
    )


print("\n==========================================")
print("GINI INDEX OF FEATURES")
print("==========================================")


features = [
    "age",
    "clinical.T.Stage",
    "Clinical.N.Stage",
    "Clinical.M.Stage",
    "Overall.Stage",
    "Histology",
    "gender"
]


for feature in features:

    if pd.api.types.is_numeric_dtype(df[feature]):

        feature_values = equal_width_binning(
            df[feature],
            4
        )

    else:

        feature_values = df[feature].fillna("Missing")

    gini = calculate_gini(feature_values)

    print(
        f"{feature:25s} : {gini:.6f}"
    )
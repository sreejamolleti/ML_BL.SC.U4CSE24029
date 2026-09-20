# ============================================================
# A3 - ROOT NODE DETECTION USING INFORMATION GAIN
# Dataset: NSCLC-Radiomics Lung1 Clinical Dataset
# ============================================================

import pandas as pd
import numpy as np
import math


CLINICAL_FILE = "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"

TARGET = "deadstatus.event"


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

df = pd.read_csv(CLINICAL_FILE)


# ------------------------------------------------------------
# Entropy Function
# ------------------------------------------------------------

def entropy(values):

    values = pd.Series(values).dropna()

    probabilities = values.value_counts(normalize=True)

    result = 0.0

    for p in probabilities:

        if p > 0:
            result -= p * math.log2(p)

    return result


# ------------------------------------------------------------
# Equal Width Binning
# ------------------------------------------------------------

def equal_width_binning(series, n_bins=4):

    numeric_series = pd.to_numeric(
        series,
        errors="coerce"
    )

    return pd.cut(
        numeric_series,
        bins=n_bins,
        include_lowest=True
    ).astype(str)


# ------------------------------------------------------------
# Prepare Feature
# ------------------------------------------------------------

def prepare_feature(series, n_bins=4):

    if pd.api.types.is_numeric_dtype(series):

        return equal_width_binning(
            series,
            n_bins
        )

    else:

        return series.fillna("Missing").astype(str)


# ------------------------------------------------------------
# Information Gain Function
# ------------------------------------------------------------

def information_gain(feature, target):

    feature = pd.Series(feature)
    target = pd.Series(target)

    valid = feature.notna() & target.notna()

    feature = feature[valid]
    target = target[valid]

    parent_entropy = entropy(target)

    weighted_entropy = 0.0

    for value in feature.unique():

        subset_target = target[
            feature == value
        ]

        weight = len(subset_target) / len(target)

        weighted_entropy += (
            weight * entropy(subset_target)
        )

    gain = parent_entropy - weighted_entropy

    return gain


# ------------------------------------------------------------
# Detect Root Node
# ------------------------------------------------------------

def find_root_node(data, target_column):

    features = [
        column
        for column in data.columns
        if column not in [
            "PatientID",
            target_column
        ]
    ]

    results = []

    for feature in features:

        prepared = prepare_feature(
            data[feature],
            n_bins=4
        )

        gain = information_gain(
            prepared,
            data[target_column]
        )

        results.append(
            (feature, gain)
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results


# ------------------------------------------------------------
# Run Root Detection
# ------------------------------------------------------------

results = find_root_node(
    df,
    TARGET
)


# ------------------------------------------------------------
# Display Information Gain
# ------------------------------------------------------------

print("==========================================")
print("A3 - INFORMATION GAIN")
print("==========================================")

print(
    f"{'Feature':30s} {'Information Gain':>20s}"
)

print("-" * 55)

for feature, gain in results:

    print(
        f"{feature:30s} {gain:20.6f}"
    )


# ------------------------------------------------------------
# Root Node
# ------------------------------------------------------------

root_feature = results[0][0]
root_gain = results[0][1]

print("\n==========================================")
print("ROOT NODE")
print("==========================================")

print("Root Feature:", root_feature)
print("Information Gain:", root_gain)


# ------------------------------------------------------------
# Explanation
# ------------------------------------------------------------

print("\nThe feature having the highest Information Gain")
print("is selected as the root node of the Decision Tree.")
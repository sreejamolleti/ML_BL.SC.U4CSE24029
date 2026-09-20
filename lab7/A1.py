
import pandas as pd
import numpy as np
import math


CLINICAL_FILE = "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
MANIFEST_FILE = "NSCLC-Radiomics-Version-4-Oct-2020-NBIA-manifest(1).tcia"


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

df = pd.read_csv(CLINICAL_FILE)

print("Clinical Dataset Shape:", df.shape)
print("Clinical Dataset Columns:")
print(df.columns.tolist())

print("\nTarget Distribution:")
print(df["deadstatus.event"].value_counts())


# ------------------------------------------------------------
# Equal Width Binning Function
# ------------------------------------------------------------

def equal_width_binning(data, column, n_bins=4):
    """
    Performs equal-width binning on a numerical attribute.

    Parameters:
        data     : pandas DataFrame
        column   : numerical column name
        n_bins   : number of bins

    Returns:
        Binned categorical Series
    """

    values = pd.to_numeric(data[column], errors="coerce")

    min_value = values.min()
    max_value = values.max()

    width = (max_value - min_value) / n_bins

    print("\nEqual Width Binning")
    print("Feature:", column)
    print("Minimum:", min_value)
    print("Maximum:", max_value)
    print("Number of bins:", n_bins)
    print("Bin width:", width)

    bins = pd.cut(
        values,
        bins=n_bins,
        include_lowest=True
    )

    return bins


# ------------------------------------------------------------
# Entropy Function
# ------------------------------------------------------------

def calculate_entropy(values):
    """
    Calculates Shannon entropy.

    H = - Sum(pi * log2(pi))
    """

    values = pd.Series(values).dropna()

    probabilities = values.value_counts(normalize=True)

    entropy = 0.0

    for probability in probabilities:
        entropy -= probability * math.log2(probability)

    return entropy


# ------------------------------------------------------------
# Entropy of Target Variable
# ------------------------------------------------------------

target = df["deadstatus.event"]

entropy_value = calculate_entropy(target)

print("\n==========================================")
print("A1 - ENTROPY")
print("==========================================")

print("Target variable: deadstatus.event")
print("Entropy =", entropy_value)


# ------------------------------------------------------------
# Demonstrate Equal Width Binning
# ------------------------------------------------------------

binned_age = equal_width_binning(
    df,
    "age",
    4
)

print("\nAge after Equal Width Binning:")
print(binned_age.value_counts().sort_index())


# ------------------------------------------------------------
# Calculate entropy of binned age
# ------------------------------------------------------------

age_entropy = calculate_entropy(binned_age)

print("\nEntropy of binned Age =", age_entropy)


# ------------------------------------------------------------
# Display target probabilities
# ------------------------------------------------------------

print("\nTarget Probabilities:")

target_probabilities = target.value_counts(normalize=True)

for value, probability in target_probabilities.items():
    print(
        "Outcome =", value,
        "Probability =", probability
    )


# ------------------------------------------------------------
# Manifest information
# ------------------------------------------------------------

try:

    with open(MANIFEST_FILE, "r", encoding="utf-8", errors="ignore") as file:
        manifest_content = file.read()

    print("\nTCIA Manifest loaded successfully.")
    print("Manifest size:", len(manifest_content), "characters")

except FileNotFoundError:

    print("\nTCIA Manifest file was not found in the current folder.")
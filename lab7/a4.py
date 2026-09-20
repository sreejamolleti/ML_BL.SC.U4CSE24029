# ============================================================
# A4 - BINNING MODULE
# Equal Width + Equal Frequency Binning
# Dataset: NSCLC-Radiomics Lung1 Clinical Dataset
# ============================================================

import pandas as pd
import numpy as np


# ------------------------------------------------------------
# DATASET FILE
# ------------------------------------------------------------

CLINICAL_FILE = "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"


# ------------------------------------------------------------
# BINNING FUNCTION
# ------------------------------------------------------------

def bin_feature(series, binning_type="equal_width", n_bins=4):
    """
    Performs binning on a numerical attribute.

    Parameters
    ----------
    series : pandas Series
        Numerical data to be binned.

    binning_type : str
        "equal_width" or "equal_frequency"

    n_bins : int
        Number of bins.

    Returns
    -------
    pandas Series
        Binned data.
    """

    # Check number of bins
    if n_bins < 2:
        raise ValueError(
            "Number of bins must be at least 2."
        )

    # Convert values to numeric
    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Equal Width Binning
    # --------------------------------------------------------

    if binning_type.lower() == "equal_width":

        result = pd.cut(
            series,
            bins=n_bins,
            labels=[
                f"Bin_{i + 1}"
                for i in range(n_bins)
            ],
            include_lowest=True
        )

    # --------------------------------------------------------
    # Equal Frequency Binning
    # --------------------------------------------------------

    elif binning_type.lower() == "equal_frequency":

        result = pd.qcut(
            series,
            q=n_bins,
            labels=[
                f"Bin_{i + 1}"
                for i in range(n_bins)
            ],
            duplicates="drop"
        )

    # --------------------------------------------------------
    # Invalid Binning Type
    # --------------------------------------------------------

    else:

        raise ValueError(
            "binning_type must be "
            "'equal_width' or 'equal_frequency'"
        )

    return result


# ------------------------------------------------------------
# DEFAULT BINNING FUNCTION
# ------------------------------------------------------------

def default_binning(series):
    """
    Performs equal-width binning using default
    parameters.

    Default:
        Binning type = equal_width
        Number of bins = 4
    """

    return bin_feature(
        series,
        binning_type="equal_width",
        n_bins=4
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("==========================================")
    print("A4 - BINNING")
    print("==========================================")

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            CLINICAL_FILE
        )

        print(
            "\nDataset loaded successfully."
        )

        print(
            "Dataset Shape:",
            df.shape
        )

    except FileNotFoundError:

        print(
            "\nERROR: Dataset file not found."
        )

        print(
            "Expected file:"
        )

        print(
            CLINICAL_FILE
        )

        exit()


    # --------------------------------------------------------
    # Check Age Column
    # --------------------------------------------------------

    if "age" not in df.columns:

        print(
            "\nERROR: 'age' column not found."
        )

        print(
            "Available columns:"
        )

        print(
            df.columns.tolist()
        )

        exit()


    # ========================================================
    # 1. EQUAL WIDTH BINNING
    # ========================================================

    print("\n==========================================")
    print("1. EQUAL WIDTH BINNING")
    print("==========================================")


    age_equal_width = bin_feature(
        df["age"],
        binning_type="equal_width",
        n_bins=4
    )


    print(
        "\nAge divided into 4 equal-width bins:"
    )


    print(
        age_equal_width.value_counts(
            sort=False
        )
    )


    # --------------------------------------------------------
    # Equal Width Bin Details
    # --------------------------------------------------------

    print(
        "\nEqual Width Bin Intervals:"
    )


    numeric_age = pd.to_numeric(
        df["age"],
        errors="coerce"
    )


    equal_width_intervals = pd.cut(
        numeric_age,
        bins=4,
        include_lowest=True
    )


    print(
        equal_width_intervals.cat.categories
    )


    # ========================================================
    # 2. EQUAL FREQUENCY BINNING
    # ========================================================

    print("\n==========================================")
    print("2. EQUAL FREQUENCY BINNING")
    print("==========================================")


    age_equal_frequency = bin_feature(
        df["age"],
        binning_type="equal_frequency",
        n_bins=4
    )


    print(
        "\nAge divided into 4 equal-frequency bins:"
    )


    print(
        age_equal_frequency.value_counts(
            sort=False
        )
    )


    # ========================================================
    # 3. DEFAULT BINNING
    # ========================================================

    print("\n==========================================")
    print("3. DEFAULT BINNING")
    print("==========================================")


    default_result = default_binning(
        df["age"]
    )


    print(
        "\nDefault binning:"
    )


    print(
        default_result.value_counts(
            sort=False
        )
    )


    # ========================================================
    # 4. COMPARE RESULTS
    # ========================================================

    print("\n==========================================")
    print("4. COMPARISON")
    print("==========================================")


    comparison = pd.DataFrame({

        "Age":
            df["age"],

        "Equal_Width":
            age_equal_width,

        "Equal_Frequency":
            age_equal_frequency

    })


    print(
        comparison.head(20).to_string(
            index=False
        )
    )


    # ========================================================
    # 5. SAVE BINNED DATA
    # ========================================================

    output_file = "A4_binned_age_results.csv"


    comparison.to_csv(
        output_file,
        index=False
    )


    print(
        "\nBinned data saved as:"
    )

    print(
        output_file
    )


    # ========================================================
    # COMPLETION MESSAGE
    # ========================================================

    print("\n==========================================")
    print("A4 COMPLETED SUCCESSFULLY")
    print("==========================================")
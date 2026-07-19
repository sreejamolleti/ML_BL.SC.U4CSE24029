import pandas as pd
import numpy as np
import os

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

    return pd.read_excel(
        file_path,
        sheet_name="thyroid0387_UCI",
        na_values=["?"]
    )

def main():

    df = load_data()

    # ================= DEBUGGING =================
    print("========== DATA TYPES ==========\n")
    print(df.dtypes)

    print("\n========== UNIQUE VALUES IN FIRST 15 COLUMNS ==========\n")

    for col in df.columns[:15]:
        print(f"{col} : {df[col].dropna().unique()[:10]}")

    print("\n========== ALL COLUMN UNIQUE VALUES ==========\n")

    for col in df.columns:
        values = set(df[col].dropna().astype(str).str.strip().str.lower())
        print(f"{col} -> {values}")

    # =============================================

    binary_columns = []

    for col in df.columns:
        values = set(df[col].dropna().astype(str).str.strip().str.lower())

        if values.issubset({"t", "f"}):
            binary_columns.append(col)

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"t": 1, "f": 0})
            )

    print("\n========== BINARY COLUMNS ==========\n")
    print(binary_columns)

    if len(binary_columns) == 0:
        print("\nNo binary columns detected.")
        return

    binary_data = df[binary_columns]

    vector1 = binary_data.iloc[0].astype(int).values
    vector2 = binary_data.iloc[1].astype(int).values

    f11 = np.sum((vector1 == 1) & (vector2 == 1))
    f10 = np.sum((vector1 == 1) & (vector2 == 0))
    f01 = np.sum((vector1 == 0) & (vector2 == 1))
    f00 = np.sum((vector1 == 0) & (vector2 == 0))

    jaccard = f11 / (f11 + f10 + f01)
    smc = (f11 + f00) / (f11 + f10 + f01 + f00)

    print("\n========== RESULTS ==========\n")

    print("f11 =", f11)
    print("f10 =", f10)
    print("f01 =", f01)
    print("f00 =", f00)

    print("\nJaccard Coefficient :", jaccard)
    print("Simple Matching Coefficient :", smc)


if __name__ == "__main__":
    main()
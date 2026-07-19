import pandas as pd
import numpy as np
import os

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def main():

    df = load_data()

    print("\n========== FIRST 5 RECORDS ==========\n")
    print(df.head())

    print("\n========== DATA TYPES ==========\n")
    print(df.dtypes)

    print("\n========== DATA SHAPE ==========\n")
    print("Rows :", df.shape[0])
    print("Columns :", df.shape[1])

    print("\n========== MISSING VALUES ==========\n")
    print(df.isnull().sum())

    print("\n========== NUMERIC DATA SUMMARY ==========\n")
    print(df.describe())

    print("\n========== MEAN ==========\n")
    print(df.mean(numeric_only=True))

    print("\n========== VARIANCE ==========\n")
    print(df.var(numeric_only=True))

    print("\n========== NUMERIC COLUMN RANGES ==========\n")

    numeric_columns = df.select_dtypes(include=np.number).columns

    for column in numeric_columns:
        print(
            f"{column} : Min = {df[column].min()}  Max = {df[column].max()}"
        )

    print("\n========== OUTLIERS (IQR METHOD) ==========\n")

    for column in numeric_columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[column] < lower) | (df[column] > upper)]

        print(column, ":", len(outliers), "outliers")

    print("\n========== CATEGORICAL COLUMNS ==========\n")

    categorical_columns = df.select_dtypes(exclude=np.number).columns

    for column in categorical_columns:
        print(column, ":", df[column].unique())


if __name__ == "__main__":
    main()
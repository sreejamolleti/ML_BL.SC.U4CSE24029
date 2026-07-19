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

    print("Missing Values Before Imputation:\n")
    print(df.isnull().sum())

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = ((df[column] < lower) | (df[column] > upper)).sum()

            if outliers > 0:
                df[column] = df[column].fillna(df[column].median())
            else:
                df[column] = df[column].fillna(df[column].mean())

        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    print("\nMissing Values After Imputation:\n")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
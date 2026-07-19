import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os 

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")


def main():

    df = load_data()

    numeric_columns = df.select_dtypes(include=["number"]).columns

    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].mean()
    )

    scaler = MinMaxScaler()

    normalized = scaler.fit_transform(df[numeric_columns])

    normalized_df = pd.DataFrame(
        normalized,
        columns=numeric_columns
    )

    print("First 10 Rows of Normalized Data\n")

    print(normalized_df.head(10))


if __name__ == "__main__":
    main()
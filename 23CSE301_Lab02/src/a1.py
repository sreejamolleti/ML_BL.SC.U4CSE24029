import os
import pandas as pd
import numpy as np

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

    excel = pd.ExcelFile(file_path)

    print("Available Sheets:")
    print(excel.sheet_names)

    return pd.read_excel(file_path, sheet_name=excel.sheet_names[0])
def create_matrices(df):
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy()

    y = df["Payment (Rs)"].to_numpy()

    return X, y


def calculate_rank(X):
    return np.linalg.matrix_rank(X)


def calculate_product_cost(X, y):
    pseudo_inverse = np.linalg.pinv(X)
    return pseudo_inverse @ y


def main():

    # Load Dataset
    df = load_data()

    # Create Feature Matrix and Output Vector
    X, y = create_matrices(df)

    print("=" * 60)
    print("PURCHASE DATA")
    print("=" * 60)
    print(df)

    print("\n" + "=" * 60)
    print("FEATURE MATRIX (X)")
    print("=" * 60)
    print(X)

    print("\n" + "=" * 60)
    print("OUTPUT VECTOR (y)")
    print("=" * 60)
    print(y)

    # Dimensionality
    print("\nDimensionality of Vector Space :", X.shape[1])

    # Number of vectors
    print("Number of Observation Vectors :", X.shape[0])

    # Rank
    rank = calculate_rank(X)
    print("Rank of Feature Matrix :", rank)

    # Product Cost Calculation
    product_cost = calculate_product_cost(X, y)

    print("\nEstimated Cost of Each Product")
    print("-" * 40)
    print(f"Candies (#)      : Rs. {product_cost[0]:.2f}")
    print(f"Mangoes (Kg)     : Rs. {product_cost[1]:.2f}")
    print(f"Milk Packets (#) : Rs. {product_cost[2]:.2f}")


if __name__ == "__main__":
    main()
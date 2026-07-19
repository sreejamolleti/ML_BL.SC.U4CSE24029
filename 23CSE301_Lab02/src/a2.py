import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

    return pd.read_excel(file_path, sheet_name="Purchase data")

def main():
    df = load_data()

    # Create Class Label
    df["Category"] = df["Payment (Rs)"].apply(
        lambda x: "RICH" if x > 200 else "POOR"
    )

    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]
    y = df["Category"]

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    predictions = model.predict(X)

    result = df[["Customer", "Payment (Rs)", "Category"]].copy()
    result["Predicted"] = predictions

    print("\nCustomer Classification")
    print(result)

    accuracy = (predictions == y).mean()

    print(f"\nClassifier Accuracy : {accuracy * 100:.2f}%")

    print("\nDecision Rules Learned Successfully.")


if __name__ == "__main__":
    main()
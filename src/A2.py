import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

df = pd.read_excel(file_path, sheet_name="marketing_campaign")

def label_encode(column):
    unique_values = column.unique()
    mapping = {}

    for i, value in enumerate(unique_values):
        mapping[value] = i

    encoded = column.map(mapping)
    return encoded, mapping

def one_hot_encode(column):
    unique_values = column.unique()
    one_hot = pd.DataFrame()

    for value in unique_values:
        one_hot[str(value)] = (column == value).astype(int)

    return one_hot

print("Original Education Column:")
print(df["Education"].head())

label_encoded, mapping = label_encode(df["Education"])

print("\nLabel Encoded Education:")
print(label_encoded.head())

print("\nLabel Encoding Mapping:")
print(mapping)

print("\nOne Hot Encoded Education:")
one_hot = one_hot_encode(df["Education"])
print(one_hot.head())
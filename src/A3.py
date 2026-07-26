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

    return column.map(mapping)

def one_hot_encode(column):
    unique_values = column.unique()
    one_hot = pd.DataFrame()

    for value in unique_values:
        one_hot[str(value)] = (column == value).astype(int)

    return one_hot

encoded_df = df.copy()

categorical_columns = encoded_df.select_dtypes(include="object").columns

for col in categorical_columns:
    if encoded_df[col].nunique() <= 5:
        one_hot = one_hot_encode(encoded_df[col])
        one_hot.columns = [f"{col}_{c}" for c in one_hot.columns]
        encoded_df = encoded_df.drop(columns=[col])
        encoded_df = pd.concat([encoded_df, one_hot], axis=1)
    else:
        encoded_df[col] = label_encode(encoded_df[col])

print("Original Dataset Shape:", df.shape)
print("Encoded Dataset Shape:", encoded_df.shape)

print("\nEncoded Dataset:")
print(encoded_df.head())
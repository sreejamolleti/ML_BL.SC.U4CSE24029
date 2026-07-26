import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

df = pd.read_excel(file_path, sheet_name="marketing_campaign")

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nColumn Names:")
for col in df.columns:
    print(col)

print("\nData Types:")
print(df.dtypes)

print("\nUnique Values in Each Column:")
for col in df.columns:
    print(f"\n{col}")
    print(df[col].unique()[:10])
import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

df = pd.read_excel(file_path, sheet_name="marketing_campaign")

numeric_df = df.select_dtypes(include=np.number)

def my_mean(data):
    return sum(data) / len(data)

def my_variance(data):
    mean = my_mean(data)
    variance = 0
    for value in data:
        variance += (value - mean) ** 2
    return variance / len(data)

def my_std(data):
    return my_variance(data) ** 0.5

print("Mean of Each Numeric Feature\n")

for column in numeric_df.columns:
    mean = my_mean(numeric_df[column])
    print(column, ":", mean)

print("\nVariance of Each Numeric Feature\n")

for column in numeric_df.columns:
    variance = my_variance(numeric_df[column])
    print(column, ":", variance)

print("\nStandard Deviation of Each Numeric Feature\n")

for column in numeric_df.columns:
    std = my_std(numeric_df[column])
    print(column, ":", std)
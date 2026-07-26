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

print("Comparison of Mean\n")

for column in numeric_df.columns:
    myMean = my_mean(numeric_df[column])
    numpyMean = np.mean(numeric_df[column])

    print(column)
    print("My Mean      :", myMean)
    print("NumPy Mean   :", numpyMean)
    print()

print("Comparison of Standard Deviation\n")

for column in numeric_df.columns:
    myStd = my_std(numeric_df[column])
    numpyStd = np.std(numeric_df[column])

    print(column)
    print("My Std Dev   :", myStd)
    print("NumPy Std Dev:", numpyStd)
    print()
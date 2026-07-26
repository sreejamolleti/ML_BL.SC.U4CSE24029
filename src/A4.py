import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

df = pd.read_excel(file_path, sheet_name="marketing_campaign")

numeric_df = df.select_dtypes(include=np.number)

A = numeric_df.iloc[0].values
B = numeric_df.iloc[1].values

def minkowski_distance(A, B, p):
    distance = 0

    for i in range(len(A)):
        distance += abs(A[i] - B[i]) ** p

    return distance ** (1 / p)

p = int(input("Enter value of p: "))

distance = minkowski_distance(A, B, p)

if p == 1:
    print("Manhattan Distance =", distance)
elif p == 2:
    print("Euclidean Distance =", distance)
else:
    print("Minkowski Distance =", distance)
import pandas as pd
import numpy as np
from scipy.spatial.distance import minkowski
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

df = pd.read_excel(file_path, sheet_name="marketing_campaign")

numeric_df = df.select_dtypes(include=np.number)

A = numeric_df.iloc[0].values
B = numeric_df.iloc[1].values

def my_minkowski(A, B, p):
    distance = 0
    for i in range(len(A)):
        distance += abs(A[i] - B[i]) ** p
    return distance ** (1 / p)

p = int(input("Enter value of p: "))

my_distance = my_minkowski(A, B, p)
package_distance = minkowski(A, B, p)

print("\nMy Minkowski Distance :", my_distance)
print("SciPy Minkowski Distance :", package_distance)

if abs(my_distance - package_distance) < 1e-6:
    print("\nBoth results are the same.")
else:
    print("\nResults are different.")
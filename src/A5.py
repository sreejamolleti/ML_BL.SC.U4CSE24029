import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

p_values = list(range(1, 11))
distances = []

for p in p_values:
    d = minkowski_distance(A, B, p)
    distances.append(d)
    print("p =", p, "Distance =", d)

plt.plot(p_values, distances, marker='o')
plt.xlabel("Value of p")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance for p = 1 to 10")
plt.grid(True)
plt.show()
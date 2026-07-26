import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

df = pd.read_excel(file_path, sheet_name="marketing_campaign")

numeric_df = df.select_dtypes(include=np.number)

A = numeric_df.iloc[0].values
B = numeric_df.iloc[1].values

def my_dot_product(A, B):
    result = 0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result

def my_norm(A):
    result = 0
    for i in range(len(A)):
        result += A[i] ** 2
    return result ** 0.5

my_dot = my_dot_product(A, B)
numpy_dot = np.dot(A, B)

my_length_A = my_norm(A)
numpy_length_A = np.linalg.norm(A)

my_length_B = my_norm(B)
numpy_length_B = np.linalg.norm(B)

print("My Dot Product:", my_dot)
print("NumPy Dot Product:", numpy_dot)

print("\nMy Length of Vector A:", my_length_A)
print("NumPy Length of Vector A:", numpy_length_A)

print("\nMy Length of Vector B:", my_length_B)
print("NumPy Length of Vector B:", numpy_length_B)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

df = pd.read_excel(file_path, sheet_name="marketing_campaign")

feature = df["Income"].dropna()

mean = np.mean(feature)
variance = np.var(feature)

print("Feature: Income")
print("Mean:", mean)
print("Variance:", variance)

plt.hist(feature, bins=10, edgecolor="black")
plt.title("Histogram of Income")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.grid(True)

plt.show()
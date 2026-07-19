import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


def calculate_mean(data):
    return sum(data) / len(data)


def calculate_variance(data):
    mean = calculate_mean(data)
    return sum((x - mean) ** 2 for x in data) / len(data)


def execution_time(func, data):
    times = []

    for _ in range(10):
        start = time.perf_counter()
        func(data)
        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def main():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

    df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

    price = df["Price"].dropna()

    print("\nPopulation Mean (NumPy):", np.mean(price))
    print("Population Variance (NumPy):", np.var(price))

    print("\nPopulation Mean (Own Function):", calculate_mean(price))
    print("Population Variance (Own Function):", calculate_variance(price))

    numpy_mean_time = execution_time(np.mean, price)
    own_mean_time = execution_time(calculate_mean, price)

    numpy_var_time = execution_time(np.var, price)
    own_var_time = execution_time(calculate_variance, price)

    print("\nAverage Execution Time (Mean)")
    print("NumPy :", numpy_mean_time)
    print("Own   :", own_mean_time)

    print("\nAverage Execution Time (Variance)")
    print("NumPy :", numpy_var_time)
    print("Own   :", own_var_time)

    wed = df[df["Day"] == "Wed"]

    print("\nWednesday Mean Price:", np.mean(wed["Price"]))

    april = df[df["Month"] == "Apr"]

    print("April Mean Price:", np.mean(april["Price"]))

    loss_probability = (df["Chg%"] < 0).mean()

    print("\nProbability of Loss:", loss_probability)

    wed_profit = ((wed["Chg%"] > 0).sum()) / len(wed)

    print("Probability of Profit on Wednesday:", wed_profit)

    conditional = ((df["Day"] == "Wed") & (df["Chg%"] > 0)).sum() / (df["Day"] == "Wed").sum()

    print("Conditional Probability (Profit | Wednesday):", conditional)

    plt.figure(figsize=(10, 5))
    plt.scatter(df["Day"], df["Chg%"])
    plt.title("Day vs Change %")
    plt.xlabel("Day")
    plt.ylabel("Change %")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
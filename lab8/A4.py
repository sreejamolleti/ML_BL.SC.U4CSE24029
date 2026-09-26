import numpy as np
import matplotlib.pyplot as plt


def step_activation(value):
    return 1 if value >= 0 else 0


def train_perceptron(inputs, targets, learning_rate,
                     max_epochs=1000):

    weights = np.array(
        [10, 0.2, -0.75],
        dtype=float
    )

    errors = []

    for epoch in range(max_epochs):

        sse = 0

        for x, target in zip(inputs, targets):

            net = np.dot(x, weights)

            output = step_activation(net)

            error = target - output

            sse += error ** 2

            weights += learning_rate * error * x

        errors.append(sse)

        if sse <= 0.002:
            return epoch + 1, errors

    return max_epochs, errors


def main():

    inputs = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ], dtype=float)

    targets = np.array(
        [0, 0, 0, 1],
        dtype=float
    )

    learning_rates = [
        0.01,
        0.1,
        0.2,
        0.5,
        0.7,
        0.9,
        1.0
    ]

    print("=" * 60)
    print("A4 - LEARNING RATE COMPARISON")
    print("=" * 60)

    results = []

    for rate in learning_rates:

        epochs, errors = train_perceptron(
            inputs,
            targets,
            rate
        )

        results.append(
            (rate, epochs)
        )

        print(
            f"Learning Rate = {rate:<5} "
            f"Iterations = {epochs}"
        )

    rates = [item[0] for item in results]
    epochs = [item[1] for item in results]

    plt.figure(figsize=(9, 6))

    plt.plot(
        rates,
        epochs,
        marker="o"
    )

    plt.xlabel("Learning Rate")
    plt.ylabel("Number of Iterations")
    plt.title("Learning Rate vs Number of Iterations")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
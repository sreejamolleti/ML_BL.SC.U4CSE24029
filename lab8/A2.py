import numpy as np
import matplotlib.pyplot as plt


def step_activation(value):
    return 1 if value >= 0 else 0


def summation_unit(inputs, weights):
    return np.dot(inputs, weights)


def train_perceptron(inputs, targets, weights, learning_rate=0.05,
                     max_epochs=1000):

    error_history = []

    for epoch in range(max_epochs):

        sum_squared_error = 0

        for x, target in zip(inputs, targets):

            net = summation_unit(x, weights)

            output = step_activation(net)

            error = target - output

            sum_squared_error += error ** 2

            weights = weights + learning_rate * error * x

        error_history.append(sum_squared_error)

        if sum_squared_error <= 0.002:
            return weights, epoch + 1, error_history

    return weights, max_epochs, error_history


def main():

    # Bias input is the first column
    inputs = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ], dtype=float)

    targets = np.array([0, 0, 0, 1], dtype=float)

    initial_weights = np.array([
        10,
        0.2,
        -0.75
    ], dtype=float)

    learning_rate = 0.05

    weights, epochs, errors = train_perceptron(
        inputs,
        targets,
        initial_weights.copy(),
        learning_rate
    )

    print("=" * 60)
    print("A2 - PERCEPTRON LEARNING FOR AND GATE")
    print("=" * 60)

    print("\nInitial Weights:", initial_weights)
    print("Learning Rate:", learning_rate)
    print("Number of Epochs:", epochs)
    print("Final Weights:", weights)
    print("Final SSE:", errors[-1])

    print("\nPredictions:")

    for x, target in zip(inputs, targets):

        output = step_activation(
            summation_unit(x, weights)
        )

        print(
            f"Input = {x[1:].astype(int)} "
            f"Target = {int(target)} "
            f"Output = {output}"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(errors) + 1),
        errors,
        marker="."
    )

    plt.xlabel("Epoch")
    plt.ylabel("Sum Squared Error")
    plt.title("AND Gate - Error vs Epoch")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
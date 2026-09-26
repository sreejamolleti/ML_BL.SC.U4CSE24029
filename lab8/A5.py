import numpy as np
import matplotlib.pyplot as plt


def step_activation(value):
    return 1 if value >= 0 else 0


def train_xor(inputs, targets, learning_rate=0.05,
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
            return weights, epoch + 1, errors, True

    return weights, max_epochs, errors, False


def main():

    inputs = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ], dtype=float)

    # XOR truth table
    targets = np.array(
        [0, 1, 1, 0],
        dtype=float
    )

    weights, epochs, errors, converged = train_xor(
        inputs,
        targets
    )

    print("=" * 60)
    print("A5 - XOR USING SINGLE PERCEPTRON")
    print("=" * 60)

    print("\nFinal Weights:", weights)
    print("Iterations:", epochs)
    print("Final SSE:", errors[-1])
    print("Converged:", converged)

    print("\nPredictions:")

    for x, target in zip(inputs, targets):

        output = step_activation(
            np.dot(x, weights)
        )

        print(
            f"Input = {x[1:].astype(int)} "
            f"Target = {int(target)} "
            f"Output = {output}"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(errors) + 1),
        errors
    )

    plt.xlabel("Epoch")
    plt.ylabel("SSE")
    plt.title("XOR - Perceptron Error")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt


def step_activation(value):
    return 1 if value >= 0 else 0


def train_perceptron(X, y, learning_rate=0.01,
                     max_epochs=1000):

    weights = np.zeros(X.shape[1])

    errors = []

    for epoch in range(max_epochs):

        sse = 0

        for xi, target in zip(X, y):

            net = np.dot(xi, weights)

            prediction = step_activation(net)

            error = target - prediction

            sse += error ** 2

            weights += learning_rate * error * xi

        errors.append(sse)

        if sse <= 0.002:
            return weights, epoch + 1, errors

    return weights, max_epochs, errors


def main():

    # Candies, Mangoes, Milk Packets, Payment
    X = np.array([
        [20, 6, 2, 386],
        [16, 3, 6, 289],
        [27, 6, 2, 393],
        [19, 1, 2, 110],
        [24, 4, 5, 280],
        [22, 1, 5, 167],
        [15, 4, 2, 271],
        [18, 4, 2, 274],
        [21, 1, 4, 148],
        [16, 2, 4, 198]
    ], dtype=float)

    # Add bias
    X = np.c_[
        np.ones(len(X)),
        X
    ]

    # Yes = 1, No = 0
    y = np.array([
        1,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        0
    ])

    weights, epochs, errors = train_perceptron(
        X,
        y,
        learning_rate=0.0001
    )

    print("=" * 65)
    print("A6 - CUSTOMER DATA PERCEPTRON")
    print("=" * 65)

    print("\nFinal Weights:")
    print(weights)

    print("\nIterations:", epochs)
    print("Final SSE:", errors[-1])

    print("\nPredictions:")

    correct = 0

    for xi, target in zip(X, y):

        prediction = step_activation(
            np.dot(xi, weights)
        )

        if prediction == target:
            correct += 1

        print(
            f"Actual = {'Yes' if target else 'No':3} "
            f"Predicted = {'Yes' if prediction else 'No'}"
        )

    accuracy = correct / len(y)

    print(
        "\nAccuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(errors) + 1),
        errors
    )

    plt.xlabel("Epoch")
    plt.ylabel("SSE")
    plt.title("Customer Dataset - Error vs Epoch")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
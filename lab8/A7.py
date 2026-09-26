import numpy as np


def step_activation(value):
    return 1 if value >= 0 else 0


def train_perceptron(X, y, learning_rate=0.0001,
                     max_epochs=1000):

    weights = np.zeros(X.shape[1])

    for epoch in range(max_epochs):

        errors = 0

        for xi, target in zip(X, y):

            prediction = step_activation(
                np.dot(xi, weights)
            )

            error = target - prediction

            if error != 0:
                errors += 1

            weights += learning_rate * error * xi

        if errors == 0:
            break

    return weights, epoch + 1


def calculate_accuracy(X, y, weights):

    predictions = []

    for xi in X:

        predictions.append(
            step_activation(
                np.dot(xi, weights)
            )
        )

    predictions = np.array(predictions)

    return np.mean(predictions == y), predictions


def main():

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

    X = np.c_[
        np.ones(len(X)),
        X
    ]

    y = np.array([
        1, 1, 1, 0, 1,
        0, 1, 1, 0, 0
    ])

    # Perceptron
    perceptron_weights, epochs = train_perceptron(
        X,
        y
    )

    perceptron_accuracy, _ = calculate_accuracy(
        X,
        y,
        perceptron_weights
    )

    # Matrix pseudoinverse
    pseudo_weights = np.linalg.pinv(X) @ y

    pseudo_predictions_raw = X @ pseudo_weights

    pseudo_predictions = (
        pseudo_predictions_raw >= 0.5
    ).astype(int)

    pseudo_accuracy = np.mean(
        pseudo_predictions == y
    )

    print("=" * 65)
    print("A7 - PERCEPTRON VS MATRIX PSEUDOINVERSE")
    print("=" * 65)

    print("\nPerceptron Weights:")
    print(perceptron_weights)

    print("\nPerceptron Iterations:", epochs)

    print(
        "Perceptron Accuracy:",
        round(perceptron_accuracy * 100, 2),
        "%"
    )

    print("\nPseudoinverse Weights:")
    print(pseudo_weights)

    print(
        "\nPseudoinverse Accuracy:",
        round(pseudo_accuracy * 100, 2),
        "%"
    )


if __name__ == "__main__":
    main()
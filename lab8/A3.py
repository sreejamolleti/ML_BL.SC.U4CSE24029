import numpy as np
import matplotlib.pyplot as plt


def bipolar_step(value):
    return 1 if value >= 0 else -1


def sigmoid(value):
    value = np.clip(value, -500, 500)
    return 1 / (1 + np.exp(-value))


def relu(value):
    return max(0, value)


def summation(inputs, weights):
    return np.dot(inputs, weights)


def train_perceptron(inputs, targets, weights,
                     activation_function,
                     learning_rate=0.05,
                     max_epochs=1000):

    errors = []

    for epoch in range(max_epochs):

        sse = 0

        for x, target in zip(inputs, targets):

            net = summation(x, weights)

            output = activation_function(net)

            error = target - output

            sse += error ** 2

            weights = weights + learning_rate * error * x

        errors.append(sse)

        if sse <= 0.002:
            return weights, epoch + 1, errors

    return weights, max_epochs, errors


def main():

    inputs = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ], dtype=float)

    print("=" * 65)
    print("A3 - AND GATE WITH DIFFERENT ACTIVATION FUNCTIONS")
    print("=" * 65)

    configurations = [
        (
            "Bipolar Step",
            bipolar_step,
            np.array([-1, -1, -1, 1], dtype=float)
        ),
        (
            "Sigmoid",
            sigmoid,
            np.array([0, 0, 0, 1], dtype=float)
        ),
        (
            "ReLU",
            relu,
            np.array([0, 0, 0, 1], dtype=float)
        )
    ]

    plt.figure(figsize=(10, 6))

    for name, function, targets in configurations:

        initial_weights = np.array(
            [10, 0.2, -0.75],
            dtype=float
        )

        weights, epochs, errors = train_perceptron(
            inputs,
            targets,
            initial_weights,
            function,
            learning_rate=0.05
        )

        print("\nActivation:", name)
        print("Epochs:", epochs)
        print("Final Weights:", weights)
        print("Final SSE:", errors[-1])

        plt.plot(
            range(1, len(errors) + 1),
            errors,
            label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("Sum Squared Error")
    plt.title("AND Gate - Different Activation Functions")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
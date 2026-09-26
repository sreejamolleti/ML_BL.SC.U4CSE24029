import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def train_backpropagation(
    X,
    y,
    learning_rate=0.05,
    max_epochs=1000,
    seed=42
):

    rng = np.random.default_rng(seed)

    # 2 hidden neurons
    W1 = rng.uniform(-1, 1, (2, 3))

    # 1 output neuron
    W2 = rng.uniform(-1, 1, (1, 3))

    X_bias = np.c_[
        np.ones(len(X)),
        X
    ]

    errors = []

    for epoch in range(max_epochs):

        sse = 0

        for x, target in zip(X_bias, y):

            # Forward propagation
            hidden_net = W1 @ x
            hidden_output = sigmoid(hidden_net)

            hidden_bias = np.r_[
                1,
                hidden_output
            ]

            output_net = W2 @ hidden_bias
            output = sigmoid(output_net)[0]

            # Error
            error = target - output

            sse += error ** 2

            # Backpropagation
            output_delta = (
                error *
                output *
                (1 - output)
            )

            hidden_delta = (
                hidden_output *
                (1 - hidden_output) *
                W2[0, 1:] *
                output_delta
            )

            # Update output weights
            W2 += (
                learning_rate *
                output_delta *
                hidden_bias
            )[None, :]

            # Update hidden weights
            W1 += (
                learning_rate *
                hidden_delta[:, None] *
                x[None, :]
            )

        errors.append(sse)

        if sse <= 0.002:
            return W1, W2, epoch + 1, errors

    return W1, W2, max_epochs, errors


def predict(X, W1, W2):

    predictions = []

    for sample in X:

        x = np.r_[1, sample]

        hidden = sigmoid(W1 @ x)

        hidden = np.r_[1, hidden]

        output = sigmoid(
            W2 @ hidden
        )[0]

        predictions.append(output)

    return np.array(predictions)


def main():

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    y = np.array([
        0,
        0,
        0,
        1
    ], dtype=float)

    W1, W2, epochs, errors = train_backpropagation(
        X,
        y
    )

    predictions = predict(
        X,
        W1,
        W2
    )

    print("=" * 60)
    print("A8 - BACKPROPAGATION FOR AND")
    print("=" * 60)

    print("\nIterations:", epochs)
    print("Final SSE:", errors[-1])

    print("\nOutputs:")

    for x, target, output in zip(
        X,
        y,
        predictions
    ):

        print(
            f"{x.astype(int)} "
            f"Target={int(target)} "
            f"Output={output:.4f}"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(errors) + 1),
        errors
    )

    plt.xlabel("Epoch")
    plt.ylabel("SSE")
    plt.title("Backpropagation - AND Gate")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def train_network(X, y, learning_rate=0.05,
                  max_epochs=1000, seed=42):

    rng = np.random.default_rng(seed)

    W1 = rng.uniform(-1, 1, (2, 3))
    W2 = rng.uniform(-1, 1, (1, 3))

    X_bias = np.c_[
        np.ones(len(X)),
        X
    ]

    errors = []

    for epoch in range(max_epochs):

        sse = 0

        for x, target in zip(X_bias, y):

            hidden = sigmoid(W1 @ x)

            hidden_bias = np.r_[
                1,
                hidden
            ]

            output = sigmoid(
                W2 @ hidden_bias
            )[0]

            error = target - output

            sse += error ** 2

            output_delta = (
                error *
                output *
                (1 - output)
            )

            hidden_delta = (
                hidden *
                (1 - hidden) *
                W2[0, 1:] *
                output_delta
            )

            W2 += (
                learning_rate *
                output_delta *
                hidden_bias
            )[None, :]

            W1 += (
                learning_rate *
                hidden_delta[:, None] *
                x[None, :]
            )

        errors.append(sse)

        if sse <= 0.002:
            break

    return W1, W2, epoch + 1, errors


def predict(X, W1, W2):

    outputs = []

    for sample in X:

        x = np.r_[
            1,
            sample
        ]

        hidden = sigmoid(W1 @ x)

        hidden = np.r_[
            1,
            hidden
        ]

        output = sigmoid(
            W2 @ hidden
        )[0]

        outputs.append(output)

    return np.array(outputs)


def main():

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    # XOR
    y = np.array([
        0,
        1,
        1,
        0
    ], dtype=float)

    W1, W2, epochs, errors = train_network(
        X,
        y
    )

    outputs = predict(
        X,
        W1,
        W2
    )

    print("=" * 60)
    print("A9 - BACKPROPAGATION FOR XOR")
    print("=" * 60)

    print("\nIterations:", epochs)
    print("Final SSE:", errors[-1])

    print("\nPredictions:")

    for x, target, output in zip(
        X,
        y,
        outputs
    ):

        predicted = 1 if output >= 0.5 else 0

        print(
            f"Input={x.astype(int)} "
            f"Target={int(target)} "
            f"Output={output:.4f} "
            f"Predicted={predicted}"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(errors) + 1),
        errors
    )

    plt.xlabel("Epoch")
    plt.ylabel("SSE")
    plt.title("Backpropagation - XOR Gate")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
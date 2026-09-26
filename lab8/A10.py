import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def train_network(X, targets,
                  learning_rate=0.05,
                  max_epochs=1000,
                  seed=42):

    rng = np.random.default_rng(seed)

    # 2 hidden neurons
    W1 = rng.uniform(-1, 1, (2, 3))

    # 2 output neurons
    W2 = rng.uniform(-1, 1, (2, 3))

    X_bias = np.c_[
        np.ones(len(X)),
        X
    ]

    errors = []

    for epoch in range(max_epochs):

        sse = 0

        for x, target in zip(
            X_bias,
            targets
        ):

            hidden = sigmoid(W1 @ x)

            hidden_bias = np.r_[
                1,
                hidden
            ]

            output = sigmoid(
                W2 @ hidden_bias
            )

            error = target - output

            sse += np.sum(
                error ** 2
            )

            output_delta = (
                error *
                output *
                (1 - output)
            )

            hidden_delta = (
                hidden *
                (1 - hidden) *
                (W2[:, 1:].T @ output_delta)
            )

            W2 += (
                learning_rate *
                output_delta[:, None] *
                hidden_bias[None, :]
            )

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

        hidden = sigmoid(
            W1 @ x
        )

        hidden = np.r_[
            1,
            hidden
        ]

        output = sigmoid(
            W2 @ hidden
        )

        outputs.append(output)

    return np.array(outputs)


def main():

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    # AND target:
    # 0 -> [0, 1]
    # 1 -> [1, 0]

    targets = np.array([
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 0]
    ], dtype=float)

    W1, W2, epochs, errors = train_network(
        X,
        targets
    )

    outputs = predict(
        X,
        W1,
        W2
    )

    print("=" * 65)
    print("A10 - TWO OUTPUT NODE NEURAL NETWORK")
    print("=" * 65)

    print("\nIterations:", epochs)
    print("Final SSE:", errors[-1])

    print("\nResults:")

    for x, target, output in zip(
        X,
        targets,
        outputs
    ):

        predicted_class = (
            1
            if output[0] > output[1]
            else 0
        )

        print(
            f"Input={x.astype(int)} "
            f"Target={target.astype(int)} "
            f"Output={np.round(output, 4)} "
            f"Predicted={predicted_class}"
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(errors) + 1),
        errors
    )

    plt.xlabel("Epoch")
    plt.ylabel("SSE")
    plt.title("Two Output Node Network - AND")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
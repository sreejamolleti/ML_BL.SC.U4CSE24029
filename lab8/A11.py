import numpy as np

from sklearn.neural_network import MLPClassifier


def main():

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # -----------------------------
    # AND
    # -----------------------------

    y_and = np.array([
        0,
        0,
        0,
        1
    ])

    and_model = MLPClassifier(
        hidden_layer_sizes=(2,),
        activation="logistic",
        learning_rate_init=0.05,
        max_iter=1000,
        random_state=42
    )

    and_model.fit(
        X,
        y_and
    )

    and_predictions = and_model.predict(X)

    print("=" * 60)
    print("A11 - SKLEARN MLPCLASSIFIER")
    print("=" * 60)

    print("\nAND Gate:")

    for x, target, prediction in zip(
        X,
        y_and,
        and_predictions
    ):

        print(
            f"{x} "
            f"Target={target} "
            f"Predicted={prediction}"
        )

    print(
        "AND Accuracy:",
        and_model.score(X, y_and)
    )

    # -----------------------------
    # XOR
    # -----------------------------

    y_xor = np.array([
        0,
        1,
        1,
        0
    ])

    xor_model = MLPClassifier(
        hidden_layer_sizes=(2,),
        activation="logistic",
        learning_rate_init=0.05,
        max_iter=1000,
        random_state=42
    )

    xor_model.fit(
        X,
        y_xor
    )

    xor_predictions = xor_model.predict(X)

    print("\nXOR Gate:")

    for x, target, prediction in zip(
        X,
        y_xor,
        xor_predictions
    ):

        print(
            f"{x} "
            f"Target={target} "
            f"Predicted={prediction}"
        )

    print(
        "XOR Accuracy:",
        xor_model.score(X, y_xor)
    )

    print(
        "\nAND Iterations:",
        and_model.n_iter_
    )

    print(
        "XOR Iterations:",
        xor_model.n_iter_
    )


if __name__ == "__main__":
    main()
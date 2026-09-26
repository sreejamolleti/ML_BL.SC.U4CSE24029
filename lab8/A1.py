import numpy as np


# -----------------------------
# Summation Unit
# -----------------------------
def summation_unit(inputs, weights, bias=0):
    inputs = np.array(inputs, dtype=float)
    weights = np.array(weights, dtype=float)

    return np.dot(inputs, weights) + bias


# -----------------------------
# Activation Functions
# -----------------------------
def step_activation(value):
    return 1 if value >= 0 else 0


def bipolar_step_activation(value):
    return 1 if value >= 0 else -1


def sigmoid_activation(value):
    value = np.clip(value, -500, 500)
    return 1 / (1 + np.exp(-value))


def tanh_activation(value):
    return np.tanh(value)


def relu_activation(value):
    return max(0, value)


def leaky_relu_activation(value, alpha=0.01):
    return value if value >= 0 else alpha * value


# -----------------------------
# Comparator / Error
# -----------------------------
def comparator_error(actual, predicted):
    return actual - predicted


def main():

    inputs = [1, 0]
    weights = [0.5, -0.2]
    bias = 1

    net = summation_unit(inputs, weights, bias)

    print("=" * 55)
    print("A1 - PERCEPTRON BASIC FUNCTIONS")
    print("=" * 55)

    print("\nInputs:", inputs)
    print("Weights:", weights)
    print("Bias:", bias)
    print("Summation:", net)

    print("\nActivation Functions:")
    print("Step:", step_activation(net))
    print("Bipolar Step:", bipolar_step_activation(net))
    print("Sigmoid:", sigmoid_activation(net))
    print("Tanh:", tanh_activation(net))
    print("ReLU:", relu_activation(net))
    print("Leaky ReLU:", leaky_relu_activation(net))

    actual = 1
    predicted = step_activation(net)

    print("\nActual:", actual)
    print("Predicted:", predicted)
    print("Error:", comparator_error(actual, predicted))


if __name__ == "__main__":
    main()
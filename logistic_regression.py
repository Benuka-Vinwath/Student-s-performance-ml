import numpy as np

# Sigmoid function
def sigmoid(z):
    # exp(-|z|) never overflows, unlike exp(-z) for large negative z
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


if __name__ == "__main__":
    values = np.array([-5, -2, 0, 2, 5])
    print("Input Values:")
    print(values)
    print("\nSigmoid Outputs:")
    print(sigmoid(values))

    # Self-check: matches the naive formula and survives extreme inputs
    assert np.allclose(sigmoid(values), 1 / (1 + np.exp(-values)))
    assert np.allclose(sigmoid(np.array([-1000.0, 1000.0])), [0.0, 1.0])

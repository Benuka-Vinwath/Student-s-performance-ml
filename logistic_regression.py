import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Test sigmoid function
values = np.array([-5, -2, 0, 2, 5])

results = sigmoid(values)

print("Input Values:")
print(values)

print("\nSigmoid Outputs:")
print(results)
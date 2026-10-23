import numpy as np

# Dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)

Y = np.array([35, 40, 50, 55, 65, 70, 80, 85], dtype=float)


# Prediction function
def predict(X, w, b):
    return w * X + b


# Cost function: Mean Squared Error
def compute_cost(X, Y, w, b):
    predictions = predict(X, w, b)
    errors = Y - predictions

    return np.mean(errors ** 2)


# Gradient descent
def gradient_descent(X, Y, w, b, learning_rate, iterations):

    n = len(X)
    cost_history = []

    for i in range(iterations):

        # Make predictions
        predictions = predict(X, w, b)

        # Calculate errors
        errors = Y - predictions

        # Calculate gradients
        dw = (-2 / n) * np.sum(X * errors)
        db = (-2 / n) * np.sum(errors)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        cost = compute_cost(X, Y, w, b)
        cost_history.append(cost)

        # Display progress every 100 iterations
        if i % 100 == 0:
            cost = compute_cost(X, Y, w, b)

            print(
                f"Iteration {i}: "
                f"w={w:.4f}, "
                f"b={b:.4f}, "
                f"cost={cost:.4f}"
            )

    return w, b,cost_history


# Initial parameters
w = 0.0
b = 0.0

# Train the model
w, b ,cost_history= gradient_descent(
    X,
    Y,
    w,
    b,
    learning_rate=0.01,
    iterations=2000
)

# Display final results
print("\nFinal parameters:")
print("Weight:", w)
print("Bias:", b)

print("\nFinal cost:")
print(compute_cost(X, Y, w, b))
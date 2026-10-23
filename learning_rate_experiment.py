import numpy as np
import matplotlib.pyplot as plt

from gradient_descent import gradient_descent

X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
Y = np.array([35, 40, 50, 55, 65, 70, 80, 85], dtype=float)

learning_rates = [0.001, 0.01]

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    w, b, cost_history = gradient_descent(
        X,
        Y,
        w=0.0,
        b=0.0,
        learning_rate=lr,
        iterations=2000
    )

    plt.plot(
        range(len(cost_history)),
        cost_history,
        label=f"Learning rate = {lr}"
    )

    print(f"\nLearning rate: {lr}")
    print(f"Final weight: {w:.4f}")
    print(f"Final bias: {b:.4f}")
    print(f"Final MSE: {cost_history[-1]:.4f}")

plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent: Learning Rate Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
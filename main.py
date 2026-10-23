import numpy as np
import matplotlib.pyplot as plt

from gradient_descent import (
    predict,
    compute_cost,
    gradient_descent
)


# ==========================================
# 1. DATASET
# ==========================================

X = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8],
    dtype=float
)

Y = np.array(
    [35, 40, 50, 55, 65, 70, 80, 85],
    dtype=float
)


print("=" * 55)
print("STUDENT PERFORMANCE PREDICTION")
print("=" * 55)

print("\nNumber of observations:", len(X))

print("Study hours:", X)

print("Exam scores:", Y)


# ==========================================
# 2. MODEL 1: MANUAL PARAMETERS
# ==========================================

w1 = 5.0
b1 = 30.0

cost1 = compute_cost(X, Y, w1, b1)


# ==========================================
# 3. MODEL 2: DIFFERENT MANUAL PARAMETERS
# ==========================================

w2 = 7.0
b2 = 28.0

cost2 = compute_cost(X, Y, w2, b2)


# ==========================================
# 4. MODEL 3: GRADIENT DESCENT
# ==========================================

print("\n" + "=" * 55)
print("GRADIENT DESCENT TRAINING")
print("=" * 55)

w3, b3, cost_history = gradient_descent(
    X,
    Y,
    w=0.0,
    b=0.0,
    learning_rate=0.01,
    iterations=2000
)

cost3 = compute_cost(X, Y, w3, b3)


# ==========================================
# 5. ANALYTICAL SOLUTION
# ==========================================

x_mean = np.mean(X)

y_mean = np.mean(Y)


# Calculate the optimal weight
numerator = np.sum(
    (X - x_mean) * (Y - y_mean)
)

denominator = np.sum(
    (X - x_mean) ** 2
)

w_optimal = numerator / denominator


# Calculate the optimal bias
b_optimal = y_mean - w_optimal * x_mean


# Calculate the optimal cost
optimal_cost = compute_cost(
    X,
    Y,
    w_optimal,
    b_optimal
)


# ==========================================
# 6. DISPLAY MODEL COMPARISON
# ==========================================

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

print(
    f"\nModel 1:"
    f"\nWeight = {w1:.4f}"
    f"\nBias = {b1:.4f}"
    f"\nMSE = {cost1:.4f}"
)

print(
    f"\nModel 2:"
    f"\nWeight = {w2:.4f}"
    f"\nBias = {b2:.4f}"
    f"\nMSE = {cost2:.4f}"
)

print(
    f"\nGradient Descent Model:"
    f"\nWeight = {w3:.4f}"
    f"\nBias = {b3:.4f}"
    f"\nMSE = {cost3:.4f}"
)

print(
    f"\nAnalytical Solution:"
    f"\nWeight = {w_optimal:.4f}"
    f"\nBias = {b_optimal:.4f}"
    f"\nMSE = {optimal_cost:.4f}"
)


# ==========================================
# 7. COMPARE PARAMETERS
# ==========================================

print("\n" + "=" * 55)
print("GRADIENT DESCENT VS ANALYTICAL SOLUTION")
print("=" * 55)

print(
    f"\nWeight difference: "
    f"{abs(w3 - w_optimal):.6f}"
)

print(
    f"Bias difference: "
    f"{abs(b3 - b_optimal):.6f}"
)

print(
    f"MSE difference: "
    f"{abs(cost3 - optimal_cost):.6f}"
)


# ==========================================
# 8. PREDICT A NEW STUDENT'S SCORE
# ==========================================

study_hours = 5.5

predicted_score = predict(
    study_hours,
    w3,
    b3
)

print("\n" + "=" * 55)
print("NEW STUDENT PREDICTION")
print("=" * 55)

print("Study hours:", study_hours)

print(
    f"Predicted exam score: "
    f"{predicted_score:.2f}"
)


# ==========================================
# 9. GRAPH 1: MODEL COMPARISON
# ==========================================

plt.figure(figsize=(10, 6))

# Actual observations
plt.scatter(
    X,
    Y,
    label="Actual Data",
    s=80
)

# Model 1
plt.plot(
    X,
    predict(X, w1, b1),
    label="Model 1",
    linewidth=2
)

# Model 2
plt.plot(
    X,
    predict(X, w2, b2),
    label="Model 2",
    linewidth=2
)

# Gradient descent model
plt.plot(
    X,
    predict(X, w3, b3),
    label="Gradient Descent Model",
    linewidth=2
)

# Analytical solution
plt.plot(
    X,
    predict(X, w_optimal, b_optimal),
    linestyle="--",
    label="Analytical Solution",
    linewidth=2
)

# New student's prediction
plt.scatter(
    [study_hours],
    [predicted_score],
    marker="X",
    s=150,
    label="New Student Prediction"
)

plt.xlabel("Study Hours")

plt.ylabel("Exam Score")

plt.title("Comparing Linear Regression Models")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ==========================================
# 10. GRAPH 2: COMPLETE LEARNING CURVE
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(cost_history)

plt.xlabel("Iteration")

plt.ylabel("Mean Squared Error")

plt.title("Gradient Descent Learning Curve")

plt.ylim(bottom=0)

plt.grid(True)

plt.tight_layout()

plt.show()



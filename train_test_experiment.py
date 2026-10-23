import numpy as np

from sklearn.model_selection import train_test_split

from gradient_descent import (
    predict,
    compute_cost,
    gradient_descent
)
from sklearn.metrics import(mean_absolute_error, mean_squared_error, r2_score)

# Dataset
X = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8],
    dtype=float
)

Y = np.array(
    [35, 40, 50, 55, 65, 70, 80, 85],
    dtype=float
)


# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=42
)


# Display the split
print("Training study hours:", X_train)
print("Training scores:", Y_train)

print("\nTesting study hours:", X_test)
print("Testing scores:", Y_test)


# Train using ONLY the training data
w, b, cost_history = gradient_descent(
    X_train,
    Y_train,
    w=0.0,
    b=0.0,
    learning_rate=0.01,
    iterations=2000
)


# Calculate training predictions
train_predictions = predict(X_train, w, b)

train_mse = compute_cost(
    X_train,
    Y_train,
    w,
    b
)


# Calculate testing predictions
test_predictions = predict(X_test, w, b)

test_mse = compute_cost(
    X_test,
    Y_test,
    w,
    b
)

# ==========================================
# UNDERFITTING VS TRAINED MODEL
# ==========================================

# Model A: Deliberately poor parameters
w_underfit = 2.0
b_underfit = 20.0

# Model B: Train using gradient descent
w_trained, b_trained, cost_history = gradient_descent(
    X_train,
    Y_train,
    w=0.0,
    b=0.0,
    learning_rate=0.01,
    iterations=2000
)

# Predictions for Model A
underfit_train_predictions = predict(
    X_train,
    w_underfit,
    b_underfit
)

underfit_test_predictions = predict(
    X_test,
    w_underfit,
    b_underfit
)

# Predictions for Model B
trained_train_predictions = predict(
    X_train,
    w_trained,
    b_trained
)

trained_test_predictions = predict(
    X_test,
    w_trained,
    b_trained
)

# Calculate MSE for each model
underfit_train_mse = compute_cost(
    X_train,
    Y_train,
    w_underfit,
    b_underfit
)

underfit_test_mse = compute_cost(
    X_test,
    Y_test,
    w_underfit,
    b_underfit
)

trained_train_mse = compute_cost(
    X_train,
    Y_train,
    w_trained,
    b_trained
)

trained_test_mse = compute_cost(
    X_test,
    Y_test,
    w_trained,
    b_trained
)

# Display comparison
print("\n========== MODEL COMPARISON ==========")

print("\nModel A: Poorly chosen parameters")

print(f"Training MSE: {underfit_train_mse:.4f}")
print(f"Testing MSE:  {underfit_test_mse:.4f}")

print("\nModel B: Gradient descent")

print(f"Training MSE: {trained_train_mse:.4f}")
print(f"Testing MSE:  {trained_test_mse:.4f}")
# ==========================================
# EVALUATION METRICS
# ==========================================

mae = mean_absolute_error(Y_test, test_predictions)

mse = mean_squared_error(Y_test, test_predictions)

rmse = np.sqrt(mse)

r2 = r2_score(Y_test, test_predictions)

print("\n========== TEST EVALUATION METRICS ==========")

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# Display results
print("\n========== RESULTS ==========")

print(f"Learned weight: {w:.4f}")
print(f"Learned bias: {b:.4f}")

print(f"\nTraining MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")


# Display individual test predictions
print("\n========== TEST PREDICTIONS ==========")

for hours, actual, predicted in zip(
    X_test,
    Y_test,
    test_predictions
):
    print(
        f"Study hours: {hours:.1f} | "
        f"Actual score: {actual:.1f} | "
        f"Predicted score: {predicted:.2f}"
    )
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. CREATE THE DATASET
# ==========================================

# Features: Study Hours, Sleep Hours, Practice Hours
X = np.array([
    [2, 6, 1],
    [3, 7, 2],
    [4, 6, 3],
    [5, 8, 2],
    [6, 7, 4],
    [7, 8, 3],
    [8, 7, 5],
    [9, 8, 4]
], dtype=float)

# Target: Exam Scores
Y = np.array([40, 48, 55, 65, 75, 80, 90, 95], dtype=float)


print("========== DATASET ==========")
print("Input Features:")
print(X)

print("\nActual Exam Scores:")
print(Y)

print("\nDataset Shape:", X.shape)
print("Number of Students:", X.shape[0])
print("Number of Features:", X.shape[1])


# ==========================================
# 2. SPLIT DATA INTO TRAINING AND TESTING SETS
# ==========================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================

def predict(X, weights, bias):
    return np.dot(X, weights) + bias


# ==========================================
# 4. COST FUNCTION (MEAN SQUARED ERROR)
# ==========================================

def compute_cost(X, Y, weights, bias):
    predictions = predict(X, weights, bias)
    errors = predictions - Y
    squared_errors = errors ** 2

    mse = np.mean(squared_errors)

    return mse


# ==========================================
# 5. GRADIENT DESCENT
# ==========================================

def gradient_descent(X, Y, weights, bias, learning_rate, iterations):

    n = len(Y)
    cost_history = []

    for i in range(iterations):

        # Calculate predictions
        predictions = predict(X, weights, bias)

        # Calculate prediction errors
        errors = predictions - Y

        # Calculate gradients
        dw = (2 / n) * np.dot(X.T, errors)
        db = (2 / n) * np.sum(errors)

        # Update weights and bias
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        # Calculate cost after updating parameters
        cost = compute_cost(X, Y, weights, bias)

        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: MSE = {cost:.4f}")

    return weights, bias, cost_history


# ==========================================
# 6. INITIALIZE MODEL PARAMETERS
# ==========================================

weights = np.zeros(X_train.shape[1])
bias = 0.0

print("\n========== TRAINING MODEL ==========")

print("Initial Weights:", weights)
print("Initial Bias:", bias)


# ==========================================
# 7. TRAIN THE MODEL
# ==========================================

weights, bias, cost_history = gradient_descent(
    X_train,
    Y_train,
    weights,
    bias,
    learning_rate=0.001,
    iterations=2000
)


# ==========================================
# 8. DISPLAY TRAINED PARAMETERS
# ==========================================

print("\n========== TRAINED MODEL ==========")

print("Final Weights:", weights)
print("Final Bias:", bias)

print("\nModel Equation:")

print(
    f"Predicted Score = "
    f"{weights[0]:.4f} * Study Hours + "
    f"{weights[1]:.4f} * Sleep Hours + "
    f"{weights[2]:.4f} * Practice Hours + "
    f"{bias:.4f}"
)


# ==========================================
# 9. EVALUATE MODEL PERFORMANCE
# ==========================================

train_predictions = predict(X_train, weights, bias)
test_predictions = predict(X_test, weights, bias)

train_mse = compute_cost(X_train, Y_train, weights, bias)
test_mse = compute_cost(X_test, Y_test, weights, bias)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print("\n========== MODEL EVALUATION ==========")

print(f"Training MSE:  {train_mse:.4f}")
print(f"Testing MSE:   {test_mse:.4f}")

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE:  {test_rmse:.4f}")


# ==========================================
# 10. COMPARE ACTUAL AND PREDICTED TEST SCORES
# ==========================================

print("\n========== TEST PREDICTIONS ==========")

for i in range(len(Y_test)):

    actual = Y_test[i]
    predicted = test_predictions[i]
    error = abs(actual - predicted)

    print(f"\nStudent {i + 1}")
    print(f"Actual Score:    {actual:.2f}")
    print(f"Predicted Score: {predicted:.2f}")
    print(f"Absolute Error:  {error:.2f}")


# ==========================================
# 11. PREDICT A NEW STUDENT'S SCORE
# ==========================================

new_student = np.array([[5, 7, 3]], dtype=float)

new_student_prediction = predict(
    new_student,
    weights,
    bias
)

print("\n========== NEW STUDENT PREDICTION ==========")

print("Study Hours:", new_student[0][0])
print("Sleep Hours:", new_student[0][1])
print("Practice Hours:", new_student[0][2])

print(
    f"Predicted Exam Score: "
    f"{new_student_prediction[0]:.2f}"
)
# Implement Linear Regression Algorithm

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------------
# Step 1: Create the dataset
# ---------------------------------------------------------

data = {
    "Years_Experience": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    ],

    "Salary": [
        35000, 38000, 42000, 46000, 50000,
        55000, 59000, 63000, 68000, 72000,
        76000, 81000, 85000, 90000, 95000
    ]
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# Step 2: Display the dataset
# ---------------------------------------------------------

print("Dataset:")
print(df)

# ---------------------------------------------------------
# Step 3: Define independent and dependent variables
# ---------------------------------------------------------

X = df[["Years_Experience"]]
y = df["Salary"]

# ---------------------------------------------------------
# Step 4: Split the dataset into training and testing data
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ---------------------------------------------------------
# Step 5: Create the Linear Regression model
# ---------------------------------------------------------

model = LinearRegression()

# ---------------------------------------------------------
# Step 6: Train the model
# ---------------------------------------------------------

model.fit(
    X_train,
    y_train
)

# ---------------------------------------------------------
# Step 7: Display model parameters
# ---------------------------------------------------------

print("\nLinear Regression Equation:")

print(
    f"Salary = "
    f"{model.coef_[0]:.2f} × Years_Experience "
    f"+ {model.intercept_:.2f}"
)

print(
    f"\nSlope (Coefficient): "
    f"{model.coef_[0]:.2f}"
)

print(
    f"Intercept: "
    f"{model.intercept_:.2f}"
)

# ---------------------------------------------------------
# Step 8: Make predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)

print("\nActual vs Predicted Salary:")

comparison = pd.DataFrame({
    "Years_Experience": X_test["Years_Experience"].values,
    "Actual_Salary": y_test.values,
    "Predicted_Salary": y_pred.round(2)
})

print(comparison)

# ---------------------------------------------------------
# Step 9: Evaluate the model
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

print("\nModel Evaluation:")

print(
    f"Mean Absolute Error (MAE): "
    f"{mae:.2f}"
)

print(
    f"Mean Squared Error (MSE): "
    f"{mse:.2f}"
)

print(
    f"Root Mean Squared Error (RMSE): "
    f"{rmse:.2f}"
)

print(
    f"R² Score: "
    f"{r2:.4f}"
)

# ---------------------------------------------------------
# Step 10: Plot the training data and regression line
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    X_train,
    y_train,
    s=80,
    label="Training Data"
)

plt.plot(
    X_train,
    model.predict(X_train),
    linewidth=2,
    label="Regression Line"
)

plt.title("Linear Regression - Training Data")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 11: Plot actual vs predicted values
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    X_test,
    y_test,
    s=100,
    label="Actual Salary"
)

plt.scatter(
    X_test,
    y_pred,
    s=100,
    marker="x",
    label="Predicted Salary"
)

plt.title("Actual vs Predicted Salary")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 12: Predict salary for a new employee
# ---------------------------------------------------------

new_employee = pd.DataFrame({
    "Years_Experience": [12]
})

predicted_salary = model.predict(
    new_employee
)

print(
    "\nNew Employee:"
)

print(
    "Years of Experience: 12"
)

print(
    f"Predicted Salary: "
    f"{predicted_salary[0]:.2f}"
)
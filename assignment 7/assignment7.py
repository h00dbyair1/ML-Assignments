# Implement SVM Algorithm for Regression
# Support Vector Regression (SVR)

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
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
# Step 4: Split the dataset
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
# Step 5: Standardize the input and target values
# ---------------------------------------------------------

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)

X_test_scaled = X_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).ravel()

# ---------------------------------------------------------
# Step 6: Create the SVR model
# ---------------------------------------------------------

model = SVR(
    kernel="rbf",
    C=100,
    gamma="scale",
    epsilon=0.1
)

# ---------------------------------------------------------
# Step 7: Train the SVR model
# ---------------------------------------------------------

model.fit(
    X_train_scaled,
    y_train_scaled
)

# ---------------------------------------------------------
# Step 8: Make predictions
# ---------------------------------------------------------

y_pred_scaled = model.predict(
    X_test_scaled
)

# Convert predictions back to original salary scale

y_pred = y_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).ravel()

# ---------------------------------------------------------
# Step 9: Display actual and predicted values
# ---------------------------------------------------------

comparison = pd.DataFrame({
    "Years_Experience": X_test["Years_Experience"].values,
    "Actual_Salary": y_test.values,
    "Predicted_Salary": y_pred.round(2)
})

print("\nActual vs Predicted Salary:")
print(comparison)

# ---------------------------------------------------------
# Step 10: Evaluate the model
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
# Step 11: Create a smooth range for visualization
# ---------------------------------------------------------

X_plot = pd.DataFrame({
    "Years_Experience": [
        x / 10
        for x in range(10, 151)
    ]
})

X_plot_scaled = X_scaler.transform(
    X_plot
)

y_plot_scaled = model.predict(
    X_plot_scaled
)

y_plot = y_scaler.inverse_transform(
    y_plot_scaled.reshape(-1, 1)
).ravel()

# ---------------------------------------------------------
# Step 12: Plot the SVR regression curve
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    X_train["Years_Experience"],
    y_train,
    s=80,
    label="Training Data"
)

plt.scatter(
    X_test["Years_Experience"],
    y_test,
    s=100,
    marker="x",
    label="Testing Data"
)

plt.plot(
    X_plot["Years_Experience"],
    y_plot,
    linewidth=2,
    label="SVR Regression Curve"
)

plt.title("Support Vector Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 13: Predict salary for a new employee
# ---------------------------------------------------------

new_employee = pd.DataFrame({
    "Years_Experience": [12]
})

new_employee_scaled = X_scaler.transform(
    new_employee
)

new_prediction_scaled = model.predict(
    new_employee_scaled
)

new_prediction = y_scaler.inverse_transform(
    new_prediction_scaled.reshape(-1, 1)
).ravel()

print("\nNew Employee:")
print("Years of Experience: 12")

print(
    f"Predicted Salary: "
    f"{new_prediction[0]:.2f}"
)
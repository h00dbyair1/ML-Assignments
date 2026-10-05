# Apply Regularization to Avoid Overfitting
# Ridge and Lasso Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# Step 1: Create the dataset
# ---------------------------------------------------------

data = {
    "Experience": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    ],

    "Salary": [
        35000, 38000, 42000, 46000, 50000,
        55000, 59000, 63000, 68000, 72000,
        76000, 81000, 85000, 90000, 95000,
        100000, 104000, 109000, 114000, 119000
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# ---------------------------------------------------------
# Step 2: Define independent and dependent variables
# ---------------------------------------------------------

X = df[["Experience"]]
y = df["Salary"]

# ---------------------------------------------------------
# Step 3: Split the dataset
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
# Step 4: Create a high-degree polynomial model
# ---------------------------------------------------------
# A high-degree polynomial model can easily overfit
# the training data.

polynomial_degree = 8

# ---------------------------------------------------------
# Step 5: Linear Regression without regularization
# ---------------------------------------------------------

linear_model = Pipeline([
    (
        "polynomial",
        PolynomialFeatures(
            degree=polynomial_degree,
            include_bias=False
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "regression",
        LinearRegression()
    )
])

linear_model.fit(
    X_train,
    y_train
)

# ---------------------------------------------------------
# Step 6: Ridge Regression (L2 Regularization)
# ---------------------------------------------------------

ridge_model = Pipeline([
    (
        "polynomial",
        PolynomialFeatures(
            degree=polynomial_degree,
            include_bias=False
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "ridge",
        Ridge(alpha=10.0)
    )
])

ridge_model.fit(
    X_train,
    y_train
)

# ---------------------------------------------------------
# Step 7: Lasso Regression (L1 Regularization)
# ---------------------------------------------------------

lasso_model = Pipeline([
    (
        "polynomial",
        PolynomialFeatures(
            degree=polynomial_degree,
            include_bias=False
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "lasso",
        Lasso(
            alpha=10.0,
            max_iter=10000
        )
    )
])

lasso_model.fit(
    X_train,
    y_train
)

# ---------------------------------------------------------
# Step 8: Make predictions
# ---------------------------------------------------------

linear_train_pred = linear_model.predict(X_train)
linear_test_pred = linear_model.predict(X_test)

ridge_train_pred = ridge_model.predict(X_train)
ridge_test_pred = ridge_model.predict(X_test)

lasso_train_pred = lasso_model.predict(X_train)
lasso_test_pred = lasso_model.predict(X_test)

# ---------------------------------------------------------
# Step 9: Define evaluation function
# ---------------------------------------------------------

def evaluate_model(name, y_train, train_pred, y_test, test_pred):

    train_mse = mean_squared_error(
        y_train,
        train_pred
    )

    test_mse = mean_squared_error(
        y_test,
        test_pred
    )

    train_r2 = r2_score(
        y_train,
        train_pred
    )

    test_r2 = r2_score(
        y_test,
        test_pred
    )

    print(f"\n{name}")
    print("-" * 40)

    print(
        f"Training MSE: {train_mse:.2f}"
    )

    print(
        f"Testing MSE: {test_mse:.2f}"
    )

    print(
        f"Training R²: {train_r2:.4f}"
    )

    print(
        f"Testing R²: {test_r2:.4f}"
    )

# ---------------------------------------------------------
# Step 10: Evaluate all three models
# ---------------------------------------------------------

evaluate_model(
    "Linear Regression",
    y_train,
    linear_train_pred,
    y_test,
    linear_test_pred
)

evaluate_model(
    "Ridge Regression",
    y_train,
    ridge_train_pred,
    y_test,
    ridge_test_pred
)

evaluate_model(
    "Lasso Regression",
    y_train,
    lasso_train_pred,
    y_test,
    lasso_test_pred
)

# ---------------------------------------------------------
# Step 11: Display comparison table
# ---------------------------------------------------------

results = []

models = [
    ("Linear Regression", linear_train_pred, linear_test_pred),
    ("Ridge Regression", ridge_train_pred, ridge_test_pred),
    ("Lasso Regression", lasso_train_pred, lasso_test_pred)
]

for name, train_pred, test_pred in models:

    results.append({
        "Model": name,
        "Train MSE": mean_squared_error(
            y_train,
            train_pred
        ),
        "Test MSE": mean_squared_error(
            y_test,
            test_pred
        ),
        "Train R2": r2_score(
            y_train,
            train_pred
        ),
        "Test R2": r2_score(
            y_test,
            test_pred
        )
    })

results_df = pd.DataFrame(results)

print("\nModel Comparison:")
print(results_df.round(4))

# ---------------------------------------------------------
# Step 12: Generate values for a smooth graph
# ---------------------------------------------------------

X_plot = pd.DataFrame({
    "Experience": [
        x / 10
        for x in range(10, 201)
    ]
})

linear_plot = linear_model.predict(X_plot)
ridge_plot = ridge_model.predict(X_plot)
lasso_plot = lasso_model.predict(X_plot)

# ---------------------------------------------------------
# Step 13: Plot Linear Regression
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    X_train,
    y_train,
    s=70,
    label="Training Data"
)

plt.plot(
    X_plot,
    linear_plot,
    linewidth=2,
    label="Linear Regression"
)

plt.title("Polynomial Linear Regression")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 14: Plot Ridge Regression
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    X_train,
    y_train,
    s=70,
    label="Training Data"
)

plt.plot(
    X_plot,
    ridge_plot,
    linewidth=2,
    label="Ridge Regression"
)

plt.title("Ridge Regression - L2 Regularization")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 15: Plot Lasso Regression
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    X_train,
    y_train,
    s=70,
    label="Training Data"
)

plt.plot(
    X_plot,
    lasso_plot,
    linewidth=2,
    label="Lasso Regression"
)

plt.title("Lasso Regression - L1 Regularization")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 16: Compare all models
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    X_train,
    y_train,
    s=70,
    label="Training Data"
)

plt.plot(
    X_plot,
    linear_plot,
    linewidth=2,
    label="Linear Regression"
)

plt.plot(
    X_plot,
    ridge_plot,
    linewidth=2,
    label="Ridge Regression"
)

plt.plot(
    X_plot,
    lasso_plot,
    linewidth=2,
    label="Lasso Regression"
)

plt.title("Comparison of Regularization Techniques")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

plt.show()
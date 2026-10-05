# Identify the association between dependent and independent variables

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ---------------------------------------------------------
# Step 1: Create a sample dataset
# ---------------------------------------------------------

data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Attendance": [60, 65, 70, 72, 75, 80, 82, 88, 92, 95],
    "Sleep_Hours": [5, 5, 6, 6, 7, 7, 7, 8, 8, 9],
    "Previous_Score": [50, 55, 60, 62, 65, 70, 72, 78, 85, 88],
    "Final_Score": [52, 58, 63, 66, 70, 74, 78, 84, 90, 94]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# ---------------------------------------------------------
# Step 2: Define dependent and independent variables
# ---------------------------------------------------------

independent_variables = [
    "Study_Hours",
    "Attendance",
    "Sleep_Hours",
    "Previous_Score"
]

dependent_variable = "Final_Score"

# ---------------------------------------------------------
# Step 3: Calculate correlation matrix
# ---------------------------------------------------------

correlation_matrix = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation_matrix)

# ---------------------------------------------------------
# Step 4: Calculate Pearson correlation
# ---------------------------------------------------------

print("\nAssociation with Final Score:")

for variable in independent_variables:
    correlation, p_value = pearsonr(
        df[variable],
        df[dependent_variable]
    )

    print(
        f"{variable}: "
        f"Correlation = {correlation:.3f}, "
        f"P-value = {p_value:.4f}"
    )

# ---------------------------------------------------------
# Step 5: Display correlation heatmap
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 6: Display scatter plots
# ---------------------------------------------------------

for variable in independent_variables:

    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        data=df,
        x=variable,
        y=dependent_variable,
        s=80
    )

    sns.regplot(
        data=df,
        x=variable,
        y=dependent_variable,
        scatter=False
    )

    plt.title(f"{variable} vs {dependent_variable}")
    plt.xlabel(variable)
    plt.ylabel(dependent_variable)
    plt.tight_layout()
    plt.show()
# Apply Dimensionality Reduction Using PCA

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# Step 1: Create the dataset
# ---------------------------------------------------------

data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Attendance": [60, 65, 70, 72, 75, 80, 82, 88, 92, 95],
    "Assignment_Score": [55, 60, 64, 68, 70, 75, 78, 84, 89, 93],
    "Previous_Score": [50, 55, 60, 62, 65, 70, 72, 78, 85, 88],
    "Final_Score": [52, 58, 63, 66, 70, 74, 78, 84, 90, 94]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# ---------------------------------------------------------
# Step 2: Select features
# ---------------------------------------------------------

features = [
    "Study_Hours",
    "Attendance",
    "Assignment_Score",
    "Previous_Score",
    "Final_Score"
]

X = df[features]

# ---------------------------------------------------------
# Step 3: Standardize the data
# ---------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nStandardized Data:")
print(pd.DataFrame(X_scaled, columns=features))

# ---------------------------------------------------------
# Step 4: Apply PCA
# ---------------------------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# ---------------------------------------------------------
# Step 5: Create a DataFrame containing PCA results
# ---------------------------------------------------------

pca_df = pd.DataFrame(
    X_pca,
    columns=["Principal_Component_1", "Principal_Component_2"]
)

print("\nPCA Transformed Data:")
print(pca_df)

# ---------------------------------------------------------
# Step 6: Display explained variance
# ---------------------------------------------------------

print("\nExplained Variance Ratio:")

for i, variance in enumerate(
    pca.explained_variance_ratio_,
    start=1
):
    print(
        f"Principal Component {i}: "
        f"{variance:.4f} "
        f"({variance * 100:.2f}%)"
    )

total_variance = pca.explained_variance_ratio_.sum()

print(
    f"\nTotal Variance Explained by 2 Components: "
    f"{total_variance * 100:.2f}%"
)

# ---------------------------------------------------------
# Step 7: Display PCA component loadings
# ---------------------------------------------------------

loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2"],
    index=features
)

print("\nPCA Component Loadings:")
print(loadings)

# ---------------------------------------------------------
# Step 8: Plot the PCA-transformed data
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    pca_df["Principal_Component_1"],
    pca_df["Principal_Component_2"],
    s=100
)

# Add labels to each point

for i in range(len(pca_df)):
    plt.annotate(
        str(i + 1),
        (
            pca_df["Principal_Component_1"][i],
            pca_df["Principal_Component_2"][i]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.title("PCA - Dimensionality Reduction")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 9: Plot explained variance
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

components = ["PC1", "PC2"]

plt.bar(
    components,
    pca.explained_variance_ratio_ * 100
)

plt.title("Explained Variance by Principal Components")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")

plt.show()
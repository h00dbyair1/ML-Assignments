# Apply Clustering Techniques Using Unsupervised Learning
# K-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Step 1: Create the dataset
# ---------------------------------------------------------

data = {
    "Study_Hours": [
        2, 2.5, 3, 3.5, 4,
        5, 5.5, 6, 6.5, 7,
        8, 8.5, 9, 9.5, 10
    ],

    "Exam_Score": [
        45, 48, 50, 52, 55,
        60, 63, 65, 68, 70,
        78, 82, 85, 88, 92
    ]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# ---------------------------------------------------------
# Step 2: Select features for clustering
# ---------------------------------------------------------

X = df[["Study_Hours", "Exam_Score"]]

# ---------------------------------------------------------
# Step 3: Standardize the data
# ---------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# Step 4: Find the optimal number of clusters
# using the Elbow Method
# ---------------------------------------------------------

inertia = []

for k in range(1, 8):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 8),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 8))
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 5: Apply K-Means Clustering
# ---------------------------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# ---------------------------------------------------------
# Step 6: Display clustered data
# ---------------------------------------------------------

print("\nClustered Dataset:")
print(df)

# ---------------------------------------------------------
# Step 7: Display cluster centers
# ---------------------------------------------------------

cluster_centers_scaled = kmeans.cluster_centers_

cluster_centers = scaler.inverse_transform(
    cluster_centers_scaled
)

print("\nCluster Centers:")

for i, center in enumerate(cluster_centers):
    print(
        f"Cluster {i}: "
        f"Study Hours = {center[0]:.2f}, "
        f"Exam Score = {center[1]:.2f}"
    )

# ---------------------------------------------------------
# Step 8: Visualize the clusters
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

for cluster in sorted(df["Cluster"].unique()):

    cluster_data = df[df["Cluster"] == cluster]

    plt.scatter(
        cluster_data["Study_Hours"],
        cluster_data["Exam_Score"],
        s=100,
        label=f"Cluster {cluster}"
    )

# Plot cluster centers

plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    marker="X",
    s=250,
    label="Cluster Centers"
)

plt.title("K-Means Clustering")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 9: Display number of students in each cluster
# ---------------------------------------------------------

print("\nNumber of Students in Each Cluster:")

print(
    df["Cluster"]
    .value_counts()
    .sort_index()
)
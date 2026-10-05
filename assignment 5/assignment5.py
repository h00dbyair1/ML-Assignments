# Apply KNN Algorithm for Classification

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ---------------------------------------------------------
# Step 1: Create the dataset
# ---------------------------------------------------------

data = {
    "Study_Hours": [
        1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
        6, 7, 7, 8, 8, 9, 9, 10, 10, 11
    ],

    "Attendance": [
        50, 55, 58, 60, 62, 65, 68, 70, 72, 75,
        77, 80, 82, 85, 87, 90, 92, 94, 96, 98
    ],

    "Previous_Score": [
        40, 45, 48, 50, 52, 55, 58, 60, 62, 65,
        68, 70, 72, 75, 78, 80, 83, 86, 90, 93
    ],

    "Result": [
        "Fail", "Fail", "Fail", "Fail", "Fail",
        "Fail", "Fail", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass", "Pass"
    ]
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# Step 2: Display the dataset
# ---------------------------------------------------------

print("Dataset:")
print(df)

# ---------------------------------------------------------
# Step 3: Select input features and target variable
# ---------------------------------------------------------

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Score"
    ]
]

y = df["Result"]

# ---------------------------------------------------------
# Step 4: Split the dataset
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ---------------------------------------------------------
# Step 5: Standardize the features
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Step 6: Find the best value of K
# ---------------------------------------------------------

k_values = range(1, 11)
accuracies = []

for k in k_values:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(
        X_train_scaled,
        y_train
    )

    prediction = knn.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    accuracies.append(accuracy)

# ---------------------------------------------------------
# Step 7: Display accuracy for each K
# ---------------------------------------------------------

print("\nAccuracy for Different K Values:")

for k, accuracy in zip(
    k_values,
    accuracies
):
    print(
        f"K = {k}: "
        f"{accuracy * 100:.2f}%"
    )

# ---------------------------------------------------------
# Step 8: Select the best K
# ---------------------------------------------------------

best_k = k_values[
    accuracies.index(
        max(accuracies)
    )
]

print(
    f"\nBest K Value: {best_k}"
)

# ---------------------------------------------------------
# Step 9: Plot K value vs accuracy
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    accuracies,
    marker="o"
)

plt.title("KNN Accuracy for Different K Values")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.xticks(list(k_values))
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# Step 10: Create the final KNN model
# ---------------------------------------------------------

knn_model = KNeighborsClassifier(
    n_neighbors=best_k
)

knn_model.fit(
    X_train_scaled,
    y_train
)

# ---------------------------------------------------------
# Step 11: Make predictions
# ---------------------------------------------------------

y_pred = knn_model.predict(
    X_test_scaled
)

print("\nActual Values:")
print(y_test.to_numpy())

print("\nPredicted Values:")
print(y_pred)

# ---------------------------------------------------------
# Step 12: Calculate final accuracy
# ---------------------------------------------------------

final_accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nFinal Model Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

# ---------------------------------------------------------
# Step 13: Display confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# ---------------------------------------------------------
# Step 14: Display classification report
# ---------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ---------------------------------------------------------
# Step 15: Plot confusion matrix
# ---------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=knn_model.classes_
)

disp.plot()

plt.title("KNN Confusion Matrix")
plt.tight_layout()

plt.show()

# ---------------------------------------------------------
# Step 16: Predict a new student's result
# ---------------------------------------------------------

new_student = pd.DataFrame(
    {
        "Study_Hours": [7],
        "Attendance": [85],
        "Previous_Score": [78]
    }
)

new_student_scaled = scaler.transform(
    new_student
)

new_prediction = knn_model.predict(
    new_student_scaled
)

print("\nNew Student Details:")
print(new_student)

print(
    "\nPredicted Result:",
    new_prediction[0]
)
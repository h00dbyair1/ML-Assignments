# Apply Naive Bayes Classification Algorithm

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
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
# Step 3: Separate input features and target variable
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
# Step 4: Split the dataset into training and testing data
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
# Step 5: Standardize the input features
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Step 6: Create and train the Naive Bayes model
# ---------------------------------------------------------

model = GaussianNB()

model.fit(
    X_train_scaled,
    y_train
)

# ---------------------------------------------------------
# Step 7: Make predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test_scaled)

print("\nActual Values:")
print(y_test.to_numpy())

print("\nPredicted Values:")
print(y_pred)

# ---------------------------------------------------------
# Step 8: Calculate accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)

# ---------------------------------------------------------
# Step 9: Display confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

# ---------------------------------------------------------
# Step 10: Display classification report
# ---------------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# ---------------------------------------------------------
# Step 11: Visualize confusion matrix
# ---------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Step 12: Predict result for a new student
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

prediction = model.predict(
    new_student_scaled
)

probability = model.predict_proba(
    new_student_scaled
)

print("\nNew Student Details:")
print(new_student)

print(
    "\nPredicted Result:",
    prediction[0]
)

print("\nPrediction Probabilities:")

for class_name, prob in zip(
    model.classes_,
    probability[0]
):
    print(
        f"{class_name}: {prob * 100:.2f}%"
    )
import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. CREATE SAMPLE MULTIMODAL DATASET
# ============================================================

data = {
    "text": [
        "Government announces new educational scholarship program for students.",
        "Scientists confirm that the newly developed vaccine passed clinical trials.",
        "Central bank announces changes to interest rates after economic review.",
        "University launches a new artificial intelligence research laboratory.",
        "Researchers publish a study about improving renewable energy efficiency.",
        "Official weather department announces heavy rainfall in several regions.",
        "Scientists discover a completely impossible machine that creates unlimited energy.",
        "Celebrity claims drinking a special liquid gives humans superpowers.",
        "Secret government document allegedly proves that the moon is made of cheese.",
        "Online post claims that one tablet can cure every known disease.",
        "Fake report says all banks will permanently close next week.",
        "Social media message claims humans can survive without oxygen."
    ],

    "label": [
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0
    ],

    "image_path": [
        "images/news1.jpg",
        "images/news2.jpg",
        "images/news3.jpg",
        "images/news4.jpg",
        "images/news5.jpg",
        "images/news6.jpg",
        "images/news7.jpg",
        "images/news8.jpg",
        "images/news9.jpg",
        "images/news10.jpg",
        "images/news11.jpg",
        "images/news12.jpg"
    ]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

print("\nTotal News Articles:", len(df))


# ============================================================
# 2. TEXT FEATURE EXTRACTION USING TF-IDF
# ============================================================

print("\nExtracting text features...")

vectorizer = TfidfVectorizer(
    max_features=500,
    stop_words="english"
)

text_features = vectorizer.fit_transform(
    df["text"]
).toarray()

print("Text feature shape:", text_features.shape)


# ============================================================
# 3. LOAD PRETRAINED RESNET18 FOR IMAGE FEATURES
# ============================================================

print("\nLoading pretrained image model...")

weights = models.ResNet18_Weights.DEFAULT

resnet = models.resnet18(weights=weights)

# Remove final classification layer
resnet.fc = torch.nn.Identity()

resnet.eval()

transform = weights.transforms()

print("Image model loaded successfully.")


# ============================================================
# 4. EXTRACT IMAGE FEATURES
# ============================================================

def extract_image_features(image_path):

    # If image does not exist, use a zero vector.
    if not os.path.exists(image_path):

        print(
            f"Warning: {image_path} not found. "
            "Using zero image features."
        )

        return np.zeros(512)

    try:

        image = Image.open(image_path).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0)

        with torch.no_grad():

            features = resnet(image)

        return features.squeeze().numpy()

    except Exception as error:

        print(
            f"Error processing {image_path}: {error}"
        )

        return np.zeros(512)


print("\nExtracting image features...")

image_features = []

for image_path in df["image_path"]:

    features = extract_image_features(image_path)

    image_features.append(features)

image_features = np.array(image_features)

print(
    "Image feature shape:",
    image_features.shape
)


# ============================================================
# 5. COMBINE TEXT AND IMAGE FEATURES
# ============================================================

print("\nCombining text and image features...")

combined_features = np.hstack(
    (text_features, image_features)
)

print(
    "Combined feature shape:",
    combined_features.shape
)


# ============================================================
# 6. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X = scaler.fit_transform(
    combined_features
)

y = df["label"].values


# ============================================================
# 7. SPLIT DATASET
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ============================================================
# 8. TRAIN FAKE NEWS CLASSIFIER
# ============================================================

print("\nTraining Fake News classifier...")

model = LogisticRegression(
    max_iter=2000
)

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n====================================")
print("MODEL PERFORMANCE")
print("====================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake News",
            "Real News"
        ],
        zero_division=0
    )
)


# ============================================================
# 11. FUNCTION FOR NEW NEWS PREDICTION
# ============================================================

def predict_news(news_text, image_path):

    # ----------------------------------------
    # Extract text features
    # ----------------------------------------

    text_vector = vectorizer.transform(
        [news_text]
    ).toarray()


    # ----------------------------------------
    # Extract image features
    # ----------------------------------------

    image_vector = extract_image_features(
        image_path
    ).reshape(1, -1)


    # ----------------------------------------
    # Combine both modalities
    # ----------------------------------------

    combined = np.hstack(
        (text_vector, image_vector)
    )


    # ----------------------------------------
    # Scale features
    # ----------------------------------------

    combined = scaler.transform(
        combined
    )


    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    prediction = model.predict(
        combined
    )[0]

    probability = model.predict_proba(
        combined
    )[0]


    # ----------------------------------------
    # Display result
    # ----------------------------------------

    if prediction == 1:

        result = "REAL NEWS"

        confidence = probability[1] * 100

    else:

        result = "FAKE NEWS"

        confidence = probability[0] * 100


    print("\n====================================")
    print("NEW NEWS PREDICTION")
    print("====================================")

    print("News:", news_text)

    print("Prediction:", result)

    print(
        f"Confidence: {confidence:.2f}%"
    )


# ============================================================
# 12. TEST THE MODEL WITH NEW NEWS
# ============================================================

new_news = (
    "Scientists publish an official study "
    "about improving solar energy technology."
)

new_image = "images/news1.jpg"

predict_news(
    new_news,
    new_image
)
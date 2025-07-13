import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib

labels = ["General", "Hazardous", "Organic", "Recyclable"]
label_map = {name: idx for idx, name in enumerate(labels)}

DATASET_PATH = "waste_dataset"

X, y = [], []
for label_name in labels:
    folder = os.path.join(DATASET_PATH, label_name)
    if not os.path.isdir(folder):
        continue
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(("jpg", "jpeg", "png")):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path).convert("RGB").resize((128, 128))
                    img_array = np.array(img) / 255.0
                    if img_array.shape == (128, 128, 3):
                        X.append(img_array.flatten())
                        y.append(label_map[label_name])
                except (UnidentifiedImageError, Exception):
                    continue

X = np.array(X)
y = np.array(y)

print(f"Total Clean Samples: {len(X)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Train improved RandomForest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    bootstrap=True,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=labels))

# Save
joblib.dump(model, "waste_classifier.pkl")
print("✅ Model saved as waste_classifier.pkl")





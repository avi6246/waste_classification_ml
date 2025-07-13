from PIL import Image
import numpy as np
import joblib

# Load trained model
model = joblib.load("waste_classifier.pkl")

# Labels as per model training
labels = ["General", "Hazardous", "Organic", "Recyclable"]

def preprocess_image(img):
    img = img.resize((128, 128)).convert("RGB")   # ← change to 128x128
    img_array = np.array(img) / 255.0
    return img_array.flatten().reshape(1, -1)
def classify_image(img):
    try:
        processed = preprocess_image(img)
        prediction = model.predict(processed)[0]
        print("PREDICTION:", prediction)  # For debugging in terminal
        return labels[prediction]
    except Exception as e:
        print("Error:", e)
        return "Classification error"

def get_disposal_method(label):
    """Returns disposal method based on predicted label"""
    methods = {
        "General": "Dispose in general waste bin.",
        "Hazardous": "Take to a hazardous waste facility.",
        "Organic": "Use compost or organic bin.",
        "Recyclable": "Place in recycling bin."
    }
    return methods.get(label, "Check local recycling rules.")






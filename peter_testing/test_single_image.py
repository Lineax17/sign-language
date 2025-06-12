import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os

# === Pfade ===
MODEL_PATH = "models/alexnet_tuned2.h5"
IMAGE_PATH = "saved_rois/W_test.jpg"
IMAGE_SIZE = (224, 224)

# === Klassenlabels automatisch aus Model (empfohlen) ===
model = tf.keras.models.load_model(MODEL_PATH)

# Umkehre class_indices (z. B. {0: 'A', 1: 'B', ...})
class_indices = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'del': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10,
    'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'space': 20,
    'T': 21, 'U': 22, 'V': 23, 'W': 24, 'X': 25, 'Y': 26, 'Z': 27
}


index_to_label = {v: k for k, v in class_indices.items()}

# === Bild vorbereiten ===
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMAGE_SIZE)
img = img / 255.0
img = img_to_array(img)
img = np.expand_dims(img, axis=0)

# === Vorhersage ===
prediction = model.predict(img)
predicted_index = np.argmax(prediction)
predicted_label = index_to_label[predicted_index]

print(f"✅ Vorhergesagt: {predicted_label}")
print(f"🔍 Wahrscheinlichkeiten: {prediction}")
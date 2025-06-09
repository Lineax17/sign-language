import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os

# === Pfade ===
MODEL_PATH = "models/alexnet_tuned.h5"
IMAGE_PATH = "peter_testing/own_A_test.png"
IMAGE_SIZE = (224, 224)

# === Klassenlabels automatisch aus Model (empfohlen) ===
model = tf.keras.models.load_model(MODEL_PATH)

# Umkehre class_indices (z. B. {0: 'A', 1: 'B', ...})
class_indices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
                 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
                 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
                 'nothing': 27, 'space': 28}
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

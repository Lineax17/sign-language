import tensorflow as tf
import numpy as np
import cv2
import sys

# ==== KONFIGURATION ====
MODEL_PATH = "peter_testing/models/trained_model.h5"  # Pfad zum gespeicherten Modell
IMAGE_PATH = "peter_testing/A508.jpg"  # Pfad zum zu testenden Bild
IMAGE_SIZE = (224, 224)  # Muss zum Input des Modells passen
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]

# ==== BILD LADEN UND VORBEREITEN ====
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Bild konnte nicht geladen werden: {image_path}")
        sys.exit(1)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image / 255.0  # Normalisierung
    image = np.expand_dims(image, axis=0)  # Batch-Dimension hinzufügen
    return image

# ==== MODELL LADEN ====
print("🔄 Lade Modell...")
model = tf.keras.models.load_model(MODEL_PATH)

# ==== VORHERSAGE ====
image = load_and_preprocess_image(IMAGE_PATH)
prediction = model.predict(image)
predicted_class = CLASS_NAMES[np.argmax(prediction)]

# ==== AUSGABE ====
print(f"✅ Vorhergesagte Klasse: {predicted_class}")
print(f"🔍 Wahrscheinlichkeiten: {prediction}")

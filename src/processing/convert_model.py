import tensorflow as tf
from pathlib import Path

MODEL_PATH = 'models/mobilenetv2_unfreezed.h5'
TFLITE_PATH = str(Path(MODEL_PATH).with_suffix('.tflite'))

# Modell laden
model = tf.keras.models.load_model(MODEL_PATH)

# In TFLite konvertieren
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# TFLite-Modell speichern
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"✅ TFLite-Modell gespeichert unter: {TFLITE_PATH}")

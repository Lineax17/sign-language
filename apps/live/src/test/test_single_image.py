from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = REPO_ROOT / "models" / "asl_mobilenetv2_unfreezed.keras"
IMAGE_PATH = REPO_ROOT / "saved_rois" / "B.jpg"
LABELS_PATH = REPO_ROOT / "apps" / "live" / "src" / "config" / "labels.txt"
IMAGE_SIZE = (224, 224)


def load_labels(path: Path) -> list[str]:
    labels = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                labels.append(parts[1])
    return labels


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Label file not found: {LABELS_PATH}")

labels = load_labels(LABELS_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

img = cv2.imread(str(IMAGE_PATH))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMAGE_SIZE)
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
predicted_index = int(np.argmax(prediction, axis=1)[0])
predicted_label = labels[predicted_index] if predicted_index < len(labels) else f"index_{predicted_index}"

print(f"Predicted: {predicted_label}")
print(f"Probabilities: {prediction[0]}")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


REPO_ROOT = Path(__file__).resolve().parents[4]
TEST_DIR = REPO_ROOT / "data" / "asl_alphabet_test"
MODEL_PATH = REPO_ROOT / "models" / "asl_mobilenetv2_freezed.keras"
OUTPUT_PATH = REPO_ROOT / "models" / "confusion_matrix_mobilenetv2_freezed.png"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

class_names = test_ds.class_names
model = tf.keras.models.load_model(MODEL_PATH)

y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([np.argmax(batch_labels.numpy(), axis=1) for _, batch_labels in test_ds], axis=0)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(16, 16))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
ax.set_title("Confusion Matrix MobileNetV2 - Test dataset")
fig.tight_layout()
fig.savefig(OUTPUT_PATH)

print(f"Confusion matrix written to: {OUTPUT_PATH}")

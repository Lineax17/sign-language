import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------
# Klassenliste
# ---------------------------
classes = [
    "A", "B", "C", "D", "delete", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "space", "T",
    "U", "V", "W", "X", "Y", "Z"
]
def create_dataframe(image_dir):
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    data = []
    for f in files:
        for cls in classes:
            if f.startswith(cls):
                data.append({"filename": f, "class": cls})
                break
    return pd.DataFrame(data)

# ---------------------------
# Verzeichnis
# ---------------------------
test_dir = r"/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/test_data"

# ---------------------------
# DataFrame + Generator
# ---------------------------
test_df = create_dataframe(test_dir)

datagen = ImageDataGenerator(rescale=1. / 255)

test_gen = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Wichtig für richtige Zuordnung!
)

# ---------------------------
# Modell und Testdaten
# ---------------------------
model = load_model("models/keras_model.h5")

# ---------------------------
# Vorhersagen und wahre Labels
# ---------------------------
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes  # Das ist die integer-Label-Liste, korrekt sortiert

# ---------------------------
# Confusion Matrix erstellen
# ---------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

# ---------------------------
# Anzeigen
# ---------------------------
plt.figure(figsize=(16, 16))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix AlexNet Tuned2 – Testdaten")
plt.tight_layout()
plt.savefig("confusion_matrix_alexnet_tuned2.png")  # Speichern statt anzeigen
print("✅ Confusion Matrix gespeichert unter: confusion_matrix_alexnet_tuned2.png")

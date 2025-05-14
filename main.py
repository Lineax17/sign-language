import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from kerastuner.tuners import Hyperband
import os
from pathlib import Path

# Einstellungen
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 10
SEED = 467
MODEL_PATH = "models/sign_language_mobilenetv2.h5"

# Verzeichnisse
TRAIN_DIR = "data/asl_alphabet_train"
VAL_DIR = "data/asl_alphabet_val"
TEST_DIR = "data/asl_alphabet_test"

# Lade Datensätze
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=SEED
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Klassenanzahl ermitteln
num_classes = len(train_ds.class_names)

# Prefetch für Performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Modellbau für Tuning
def build_model(hp):
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())

    # Hyperparameter-Tuning für Dense-Layer
    hp_units = hp.Int("units", min_value=64, max_value=512, step=64)
    model.add(layers.Dense(units=hp_units, activation='relu'))

    # Dropout-Tuning
    hp_dropout = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)
    model.add(layers.Dropout(hp_dropout))

    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Tuner initialisieren
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='tuner_logs',
    project_name='asl_tuning'
)

# Autotuning
tuner.search(train_ds, validation_data=val_ds, epochs=EPOCHS)

# Bestes Modell holen
best_model = tuner.get_best_models(num_models=1)[0]

# Finale Trainingsdaten: train + val zusammenführen
final_train_ds = train_ds.concatenate(val_ds)

# Modell trainieren
best_model.fit(final_train_ds, epochs=EPOCHS)

# Modell evaluieren
loss, accuracy = best_model.evaluate(test_ds)
print(f"\n✅ Endgültige Testgenauigkeit: {accuracy:.2%}")

# Modell speichern
os.makedirs(Path(MODEL_PATH).parent, exist_ok=True)
best_model.save(MODEL_PATH)
print(f"✅ Modell gespeichert unter: {MODEL_PATH}")

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from keras_tuner.tuners import RandomSearch
import os

# Settings
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 15  # längeres Training für bessere Ergebnisse
TRAIN_DIR = "data/asl_alphabet_train"
TEST_DIR = "data/asl_alphabet_test"
MODEL_PATH = "models/sign_language_tuned.h5"

# Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

num_classes = len(train_ds.class_names)


test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# Hyperparameter-tunable model builder
def build_model(hp):
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(
            hp.Int('dense_units', min_value=128, max_value=1024, step=128),
            activation=hp.Choice('dense_activation', ['relu', 'swish'])
        ),
        layers.Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.6, step=0.1)),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Tuner einrichten
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,  # je höher, desto bessere Chancen auf optimales Ergebnis
    executions_per_trial=2,  # stabilisiert Auswertung
    directory='autotune',
    project_name='asl_mobilenetv2_tuning',
    overwrite=True
)

# Start Tuning
tuner.search(train_ds, validation_data=test_ds, epochs=EPOCHS)

# Bestes Modell laden
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluation
loss, accuracy = best_model.evaluate(test_ds)
print(f"\n✅ Beste Test-Genauigkeit: {accuracy:.2%}")

# Modell speichern
os.makedirs("models", exist_ok=True)
best_model.save(MODEL_PATH)
print(f"✅ Bestes Modell gespeichert unter: {MODEL_PATH}")

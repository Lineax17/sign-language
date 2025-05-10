import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os

# Settings
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = "data/asl_alphabet_train"
TEST_DIR = "data/asl_alphabet_test"
MODEL_PATH = "models/sign_language_mobilenetv2.h5"

# load train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# load test data
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# number of classes
num_classes = len(train_ds.class_names)

# load pretrained model
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# building of model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# train model
model.fit(
    train_ds,
    epochs=EPOCHS
)

# evaluation on test dataset
loss, accuracy = model.evaluate(test_ds)
print(f"\n✅ Test-Genauigkeit: {accuracy:.2%}")

# save model
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ Modell gespeichert unter: {MODEL_PATH}")

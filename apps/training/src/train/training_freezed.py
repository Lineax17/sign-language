import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# === Configuration ===
REPO_ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = REPO_ROOT / "data" / "asl_alphabet_train"
VAL_DIR = REPO_ROOT / "data" / "asl_alphabet_val"
TEST_DIR = REPO_ROOT / "data" / "asl_alphabet_test"
SAVE_MODEL_PATH = REPO_ROOT / "models" / "asl_mobilenetv2_freezed.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

SEED = 467

# === Global Tensorflow Seed ===
tf.random.set_seed(SEED)

# === Loading Data ===
def get_dataset(directory):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True
    )

train_ds = get_dataset(TRAIN_DIR)
val_ds   = get_dataset(VAL_DIR)
test_ds  = get_dataset(TEST_DIR)

# === Save Classnames ===
train_ds_raw = get_dataset(TRAIN_DIR)
class_names = train_ds_raw.class_names

# === Prefetching & Caching ===
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# === Augmentation Layer ===
data_augmentation = tf.keras.Sequential([
    RandomRotation(0.1),             # Rotate +/- 10%
    RandomZoom(0.1),                 # Zoom +/- 10%
    RandomContrast(0.1)              # Contrast adjustment (replaces brightness)
], name="augmentation_layer")

# === Model construction: Transfer Learning with MobileNetV2 ===
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # freeze lower layers

inputs = Input(shape=(224, 224, 3))
x = data_augmentation(inputs)  # Augmentation is applied during training, ignored during inference
x = tf.keras.applications.mobilenet_v2.preprocess_input(x) # MobileNetV2 Preprocessing
x = base_model(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# === Training ===
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(str(SAVE_MODEL_PATH), save_best_only=True)
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

# === Modell speichern ===
model.save(str(SAVE_MODEL_PATH))

# === Testbewertung ===
model = load_model(str(SAVE_MODEL_PATH))
loss, acc = model.evaluate(test_ds)
print(f"\n🧪 Final Test Accuracy: {acc * 100:.2f}%")
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K

# === Pfade ===
train_dir = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/train_data"
val_dir   = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/val_data"
test_dir  = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/test_data"
save_path = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/project/sign-language/models/mobilenetv2_unfreezed.h5"

# === Klassen extrahieren ===
classes = sorted({f.split('_')[0] for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))})

def create_dataframe(image_dir):
    return pd.DataFrame([
        {"filename": f, "class": f.split('_')[0]}
        for f in os.listdir(image_dir)
        if f.endswith(('.jpg', '.png')) and f.split('_')[0] in classes
    ])

train_df = create_dataframe(train_dir)
val_df   = create_dataframe(val_dir)
test_df  = create_dataframe(test_dir)

# === Vereinfachte Augmentation (ohne Grayscale oder harte Transformationen) ===
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1]
).flow_from_dataframe(
    train_df, train_dir, x_col='filename', y_col='class',
    target_size=(224, 224), class_mode='categorical', batch_size=32
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    val_df, val_dir, x_col='filename', y_col='class',
    target_size=(224, 224), class_mode='categorical', batch_size=32
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df, test_dir, x_col='filename', y_col='class',
    target_size=(224, 224), class_mode='categorical', batch_size=32, shuffle=False
)

# === MobileNetV2 laden (ohne Top)
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# === Nur letzte 30 Layer trainierbar machen
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# === Klassifikator bauen
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(classes), activation='softmax')(x)

# === Modell bauen
model = Model(inputs=base_model.input, outputs=output)

# === Kompilieren
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(save_path, save_best_only=True)
]

# === Training
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks
)

# === Modell speichern
model.save(save_path)
print(f"✅ Modell gespeichert unter: {save_path}")

# === Evaluation auf Testdaten
loss, acc = model.evaluate(test_gen)
print(f"\n🧪 Final Test Accuracy: {acc * 100:.2f}%")

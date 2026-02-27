import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# === Pfade ===
train_dir = "data/train_data"
val_dir = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/val_data"
test_dir = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/test_data"
model_path = "models/mobilenetv2_freezed.h5"

# === Klassen extrahieren ===
classes = sorted(list(set(f.split('_')[0] for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png')))))

# === DataFrame-Erstellung ===
def create_dataframe(image_dir, classes):
    return pd.DataFrame([
        {'filename': f, 'class': f.split('_')[0]}
        for f in os.listdir(image_dir)
        if f.endswith(('.jpg', '.png')) and f.split('_')[0] in classes
    ])

train_df = create_dataframe(train_dir, classes)
val_df = create_dataframe(val_dir, classes)
test_df = create_dataframe(test_dir, classes)

# === ImageDataGenerator ===
img_size = (224, 224)
batch_size = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
).flow_from_dataframe(
    train_df, train_dir, x_col='filename', y_col='class',
    target_size=img_size, class_mode='categorical', batch_size=batch_size)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    val_df, val_dir, x_col='filename', y_col='class',
    target_size=img_size, class_mode='categorical', batch_size=batch_size)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
    test_df, test_dir, x_col='filename', y_col='class',
    target_size=img_size, class_mode='categorical', batch_size=batch_size, shuffle=False)

# === Modellaufbau: Transfer Learning mit MobileNetV2 ===
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # untere Schichten einfrieren

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# === Training ===
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(model_path, save_best_only=True)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)

# === Modell speichern ===
model.save(model_path)

# === Testbewertung ===
model = load_model(model_path)
loss, acc = model.evaluate(test_gen)
print(f"\n🧪 Final Test Accuracy: {acc * 100:.2f}%")

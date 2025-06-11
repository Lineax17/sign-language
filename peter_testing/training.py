import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==== Klassenliste ====
classes = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "space"
]

# ==== Pfade ====
train_dir = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/train"
val_dir = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/val"
test_dir = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/test"
model_path = "models/alexnet_tuned2.h5"

# ==== Hilfsfunktion ====
def create_dataframe(image_dir):
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    data = []
    for f in files:
        for cls in classes:
            if f.startswith(cls):
                data.append({"filename": f, "class": cls})
                break
    return pd.DataFrame(data)

# ==== DataFrames ====
train_df = create_dataframe(train_dir)
val_df = create_dataframe(val_dir)
test_df = create_dataframe(test_dir)

# ==== Data Generators ====
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(
    train_df, train_dir, x_col='filename', y_col='class',
    target_size=(224, 224), batch_size=32, class_mode='categorical')

val_gen = datagen.flow_from_dataframe(
    val_df, val_dir, x_col='filename', y_col='class',
    target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

test_gen = datagen.flow_from_dataframe(
    test_df, test_dir, x_col='filename', y_col='class',
    target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# ==== Modelldefinition ====
model = Sequential([
    Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2), strides=(2, 2)),
    BatchNormalization(),

    Conv2D(256, (11, 11), activation='relu'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    BatchNormalization(),

    Conv2D(384, (3, 3), activation='relu'),
    BatchNormalization(),

    Conv2D(384, (3, 3), activation='relu'),
    BatchNormalization(),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2), strides=(2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),

    Dense(512, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),

    Dense(len(classes), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==== Training ====
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=12,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True)
    ]
)

# ==== Evaluation ====
model = load_model(model_path)
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

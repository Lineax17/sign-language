import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD

# ---------------------------
# Klassenliste
# ---------------------------
classes = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]


# ---------------------------
# Hilfsfunktion für DataFrame
# ---------------------------
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
# Verzeichnisse
# ---------------------------
train_dir = r"/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/train"
val_dir = r"/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/val"
test_dir = r"/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/test"

# Zielverzeichnis zum Speichern des Modells
model_save_path = r"/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/project/sign-language/peter_testing/models"
os.makedirs(model_save_path, exist_ok=True)

# ---------------------------
# DataFrames vorbereiten
# ---------------------------
train_df = create_dataframe(train_dir)
val_df = create_dataframe(val_dir)
test_df = create_dataframe(test_dir)

# ---------------------------
# ImageDataGenerator
# ---------------------------
datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_gen = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=val_dir,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_gen = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='filename',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# ---------------------------
# Modellarchitektur
# ---------------------------
model = Sequential()
model.add(Conv2D(96, (11, 11), strides=(4, 4), padding='valid', input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(256, (11, 11), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(BatchNormalization())

model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(len(classes)))
model.add(Activation('softmax'))

# ---------------------------
# Kompilierung
# ---------------------------
sgd = SGD(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# ---------------------------
# Training
# ---------------------------
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# ---------------------------
# Test-Evaluation
# ---------------------------
loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc * 100:.2f}%")

# ---------------------------
# Modell speichern
# ---------------------------
model.save(os.path.join(model_save_path, "trained_model.h5"))
print("Modell gespeichert unter:")
print(os.path.join(model_save_path, "trained_model.h5"))

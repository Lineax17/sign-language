import os
import pandas as pd
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==== Pfade ====
train_dir = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/train_data"
val_dir = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/val_data"
test_dir = "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/test_data"
model_path = "models/alexnet_tuned3.h5"

# ==== Klassen automatisch aus Dateinamen extrahieren ====
def extract_classes_from_filenames(image_dir):
    class_names = set()
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            prefix = f.split('_')[0]
            class_names.add(prefix)
    return sorted(class_names)

# ==== DataFrame-Erzeugung + Bildprüfung ====
def create_dataframe(image_dir, classes):
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    data = []
    for f in files:
        prefix = f.split('_')[0]
        if prefix in classes:
            full_path = os.path.join(image_dir, f)
            if os.path.exists(full_path):
                data.append({"filename": f, "class": prefix})
            else:
                print(f"⚠️ Datei fehlt oder beschädigt: {f}")
    return pd.DataFrame(data)

# ==== Automatische Klassenliste ====
classes = extract_classes_from_filenames(train_dir)
print(f"📦 Erkannte Klassen: {classes}")

# ==== DataFrames ====
train_df = create_dataframe(train_dir, classes)
val_df = create_dataframe(val_dir, classes)
test_df = create_dataframe(test_dir, classes)

# ==== Data Generators mit Augmentation ====
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False  # wichtig: ASL nicht spiegeln
)

train_gen = datagen.flow_from_dataframe(
    train_df, train_dir, x_col='filename', y_col='class',
    target_size=(224, 224), batch_size=32, class_mode='categorical')

val_gen = datagen.flow_from_dataframe(
    val_df, val_dir, x_col='filename', y_col='class',
    target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

test_gen = datagen.flow_from_dataframe(
    test_df, test_dir, x_col='filename', y_col='class',
    target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# ==== Modelldefinition mit Tuning ====
def build_model(hp):
    model = Sequential()

    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (11, 11), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(hp.Int('dense_1_units', min_value=512, max_value=2048, step=256), activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(hp.Int('dense_2_units', min_value=256, max_value=1024, step=128), activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Dense(len(classes), activation='softmax'))

    hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    hp_optimizer = hp.Choice('optimizer', ['adam', 'sgd'])

    if hp_optimizer == 'adam':
        optimizer = Adam(learning_rate=hp_learning_rate)
    else:
        optimizer = SGD(learning_rate=hp_learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# ==== Tuner-Konfiguration ====
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=3,
    executions_per_trial=1,
    directory='keras_tuner',
    project_name='alexnet_tuning'
)

# ==== Tuning starten ====
tuner.search(train_gen, validation_data=val_gen, epochs=30,
             callbacks=[EarlyStopping(patience=2)])

# ==== Bestes Modell speichern und evaluieren ====
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save(model_path)

# ==== Test-Evaluation ====
model = load_model(model_path)
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Test Accuracy (tuned): {acc * 100:.2f}%")

import os
import pandas as pd
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
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

# ==== Modell mit Hyperparametern ====
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

# ==== Keras Tuner Setup ====
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='keras_tuner',
    project_name='alexnet_tuning'
)

# ==== Hyperparameter Tuning ====
tuner.search(train_gen, validation_data=val_gen, epochs=10,
             callbacks=[EarlyStopping(patience=2)])

# ==== Bestes Modell laden und speichern ====
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save(model_path)

# ==== Evaluation ====
model = load_model(model_path)
loss, acc = model.evaluate(test_gen)
print(f"\n✅ Test Accuracy (tuned): {acc * 100:.2f}%")

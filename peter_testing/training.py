import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from keras_tuner.tuners import Hyperband

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
model_save_path = r"/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/project/sign-language/peter_testing/models"
os.makedirs(model_save_path, exist_ok=True)

train_df = create_dataframe(train_dir)
val_df = create_dataframe(val_dir)
test_df = create_dataframe(test_dir)

# ---------------------------
# ImageDataGenerator
# ---------------------------
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

target_size = (200, 200)
batch_size = 16  # kleiner, GPU-freundlich

train_gen = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='filename',
    y_col='class',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_gen = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=val_dir,
    x_col='filename',
    y_col='class',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_gen = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='filename',
    y_col='class',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ---------------------------
# Modellbau für KerasTuner
# ---------------------------
def build_model(hp):
    model = Sequential()

    model.add(Conv2D(
        filters=hp.Choice('conv1_filters', [32, 64]),
        kernel_size=hp.Choice('conv1_kernel', [3, 5]),
        activation='relu',
        input_shape=(200, 200, 3)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if hp.Boolean('use_second_conv'):
        model.add(Conv2D(
            filters=hp.Choice('conv2_filters', [64, 128]),
            kernel_size=3,
            activation='relu'
        ))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(
        units=hp.Choice('dense_units', [256, 512]),
        activation='relu'
    ))
    model.add(Dropout(hp.Float('dropout', 0.3, 0.6, step=0.1)))

    model.add(Dense(len(classes), activation='softmax'))

    optimizer_name = hp.Choice('optimizer', ['adam', 'sgd'])
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=hp.Choice('adam_lr', [1e-3, 1e-4]))
    else:
        optimizer = SGD(learning_rate=hp.Choice('sgd_lr', [1e-2, 1e-3]))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------
# Tuner Setup
# ---------------------------
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='asl_tuning',
    project_name='sign_language_small'
)

# ---------------------------
# Callbacks
# ---------------------------
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2),
    ModelCheckpoint(
        filepath=os.path.join(model_save_path, 'best_model_small_{epoch:02d}-{val_accuracy:.2f}.h5'),
        monitor='val_accuracy',
        save_best_only=True
    )
]

# ---------------------------
# Tuning starten
# ---------------------------
tuner.search(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks)

# ---------------------------
# Bestes Modell verwenden
# ---------------------------
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(1)[0]

print("\nBeste Hyperparameter:")
for param in best_hps.values:
    print(f"{param}: {best_hps.get(param)}")

# ---------------------------
# Modell evaluieren
# ---------------------------
loss, acc = best_model.evaluate(test_gen)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

# ---------------------------
# Modell speichern
# ---------------------------
final_model_path = os.path.join(model_save_path, f"alex_netV1.h5")
best_model.save(final_model_path)
print(f"\nFinales Modell gespeichert unter:\n{final_model_path}")

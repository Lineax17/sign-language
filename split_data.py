import random
import shutil
from pathlib import Path

# Einstellungen
SEED = 467
SPLIT_RATIOS = {
    "train": 0.5,
    "val": 0.2,
    "test": 0.3
}

# Quell- und Zielverzeichnisse
ORIG_DIR = Path("data/asl_alphabet_original")
TRAIN_DIR = Path("data/asl_alphabet_train")
VAL_DIR   = Path("data/asl_alphabet_val")
TEST_DIR  = Path("data/asl_alphabet_test")

# Zielverzeichnisse zurücksetzen
for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)

# Zufallsseed setzen
random.seed(SEED)

# Split pro Klasse
for class_dir in ORIG_DIR.iterdir():
    if class_dir.is_dir():
        images = list(class_dir.glob("*"))
        random.shuffle(images)

        total = len(images)
        n_train = int(total * SPLIT_RATIOS["train"])
        n_val   = int(total * SPLIT_RATIOS["val"])
        n_test  = total - n_train - n_val  # Rest geht in Test

        train_images = images[:n_train]
        val_images   = images[n_train:n_train + n_val]
        test_images  = images[n_train + n_val:]

        # Zielverzeichnisse erstellen
        (TRAIN_DIR / class_dir.name).mkdir(parents=True, exist_ok=True)
        (VAL_DIR   / class_dir.name).mkdir(parents=True, exist_ok=True)
        (TEST_DIR  / class_dir.name).mkdir(parents=True, exist_ok=True)

        # Dateien kopieren
        for img in train_images:
            shutil.copy(img, TRAIN_DIR / class_dir.name / img.name)
        for img in val_images:
            shutil.copy(img, VAL_DIR / class_dir.name / img.name)
        for img in test_images:
            shutil.copy(img, TEST_DIR / class_dir.name / img.name)

        print(f"✅ {class_dir.name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

print("\n✅ Split abgeschlossen.")

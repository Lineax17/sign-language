import random
import shutil
from pathlib import Path
from PIL import Image

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

        def copy_and_flip(split_images, target_dir):
            n_flip = len(split_images) // 2
            flip_images = set(random.sample(split_images, n_flip)) if n_flip > 0 else set()
            for img in split_images:
                shutil.copy(img, target_dir / class_dir.name / img.name)
                if img in flip_images:
                    with Image.open(img) as im:
                        im_flipped = im.transpose(Image.FLIP_LEFT_RIGHT)
                        flipped_name = img.stem + "_flipped" + img.suffix
                        im_flipped.save(target_dir / class_dir.name / flipped_name)
            return n_flip

        # Zielverzeichnisse erstellen
        (TRAIN_DIR / class_dir.name).mkdir(parents=True, exist_ok=True)
        (VAL_DIR   / class_dir.name).mkdir(parents=True, exist_ok=True)
        (TEST_DIR  / class_dir.name).mkdir(parents=True, exist_ok=True)

        n_flip_train = copy_and_flip(train_images, TRAIN_DIR)
        n_flip_val   = copy_and_flip(val_images, VAL_DIR)
        n_flip_test  = copy_and_flip(test_images, TEST_DIR)

        print(f"✅ {class_dir.name}: {len(train_images)} train ({n_flip_train} gespiegelt), "
              f"{len(val_images)} val ({n_flip_val} gespiegelt), "
              f"{len(test_images)} test ({n_flip_test} gespiegelt)")

print("\n✅ Split abgeschlossen.")
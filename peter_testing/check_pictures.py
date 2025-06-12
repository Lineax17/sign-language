import os
from PIL import Image

# === Ordner definieren ===
folders = {
    "train": "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/train_data",
    "val": "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/val_data",
    "test": "/mnt/c/Users/peter/THD/4_Semester/Computer_Vision/images/test_data"
}

# === Unterstützte Bildformate ===
valid_extensions = (".jpg", ".jpeg", ".png")

def check_images(folder_path):
    total = 0
    valid = 0
    broken_files = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            total += 1
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Nur prüfen, nicht laden
                valid += 1
            except Exception as e:
                broken_files.append((filename, str(e)))

    return total, valid, broken_files

# === Auswertung ===
for label, path in folders.items():
    print(f"\n🔍 Checking '{label}' folder...")
    total, valid, broken = check_images(path)
    print(f"🟢 Gültige Bilder: {valid} / {total}")
    if broken:
        print(f"❌ Fehlerhafte Dateien:")
        for name, error in broken:
            print(f"  - {name} ({error})")
    else:
        print("✅ Keine defekten Bilder gefunden.")

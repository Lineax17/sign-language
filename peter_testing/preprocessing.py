import os
import shutil
import random

# Pfade
source_base = "C:/Users/peter/THD/4_Semester/Computer_Vision/images/asl_alphabet_train"
target_base = "C:/Users/peter/THD/4_Semester/Computer_Vision/images"

# Datensatzaufteilung
sets = {
    "train": 2000,
    "val": 750,
    "test": 250
}

# Klassenliste
class_list = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q",
    "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "delete", "space"
]

# Zielordner anlegen
def create_target_dirs():
    for split in sets:
        path = os.path.join(target_base, split)
        os.makedirs(path, exist_ok=True)

# Bilder verschieben
def split_and_move_images():
    for class_name in class_list:
        source_dir = os.path.join(source_base, class_name)
        images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) < sum(sets.values()):
            print(f"[WARNUNG] {class_name}: nur {len(images)} Bilder – benötigt: {sum(sets.values())}")
            continue

        random.shuffle(images)
        start = 0

        for split, count in sets.items():
            end = start + count
            for img_file in images[start:end]:
                src = os.path.join(source_dir, img_file)
                dst = os.path.join(target_base, split, img_file)

                # Prüfen, ob Datei bereits existiert
                if os.path.exists(dst):
                    name, ext = os.path.splitext(img_file)
                    dst = os.path.join(target_base, split, f"{name}_{random.randint(10000,99999)}{ext}")
                shutil.move(src, dst)
            start = end

        print(f"[OK] {class_name} verarbeitet.")

def main():
    print("Initialisiere Zielordner...")
    create_target_dirs()
    print("Starte Verteilung und Verschiebung...")
    split_and_move_images()
    print("Fertig! Alle Bilder wurden verteilt.")

if __name__ == "__main__":
    main()

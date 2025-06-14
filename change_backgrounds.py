from pathlib import Path
from PIL import Image
import numpy as np

# Verzeichnisse
TRAIN_DIR = Path("data/asl_alphabet_train")
VAL_DIR = Path("data/asl_alphabet_val")
TEST_DIR = Path("data/asl_alphabet_test")

# Schwellenwert für "schwarze" Pixel
SCHWARZ_GRENZWERT = 30


def schwarzer_zu_weisser_hintergrund(bild_pfad):
    """Wandelt schwarzen Hintergrund in weiß um"""
    with Image.open(bild_pfad) as img:
        # In NumPy-Array umwandeln für effiziente Bearbeitung
        img_array = np.array(img)

        # Schwarze Pixel identifizieren (alle RGB-Werte unter dem Grenzwert)
        dunkle_pixel = (img_array[:, :, 0] < SCHWARZ_GRENZWERT) & \
                       (img_array[:, :, 1] < SCHWARZ_GRENZWERT) & \
                       (img_array[:, :, 2] < SCHWARZ_GRENZWERT)

        # Schwarze Pixel in weiß umwandeln
        img_array[dunkle_pixel] = [255, 255, 255]

        # Array zurück in ein Bild umwandeln und speichern
        bearbeitetes_bild = Image.fromarray(img_array)
        bearbeitetes_bild.save(bild_pfad)


# Alle Verzeichnisse durchlaufen
alle_verzeichnisse = [TRAIN_DIR, VAL_DIR, TEST_DIR]
gesamt_konvertiert = 0

for verzeichnis in alle_verzeichnisse:
    if not verzeichnis.exists():
        print(f"⚠️ Verzeichnis {verzeichnis} nicht gefunden.")
        continue

    for klassen_verzeichnis in verzeichnis.iterdir():
        if not klassen_verzeichnis.is_dir():
            continue

        klasse_konvertiert = 0
        for bild_pfad in klassen_verzeichnis.glob("*"):
            if bild_pfad.is_file() and bild_pfad.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    schwarzer_zu_weisser_hintergrund(bild_pfad)
                    klasse_konvertiert += 1
                except Exception as e:
                    print(f"❌ Fehler bei {bild_pfad}: {e}")

        print(f"✅ {klassen_verzeichnis.name} in {verzeichnis.name}: {klasse_konvertiert} Bilder konvertiert")
        gesamt_konvertiert += klasse_konvertiert

print(f"\n✅ Insgesamt {gesamt_konvertiert} Bilder mit weißem Hintergrund erstellt.")
import random
import shutil
from pathlib import Path
from typing import List, Dict

# --- Configuration ---
SEED = 467
SPLIT_RATIOS = {
    "train": 0.5,
    "val": 0.2,
    "test": 0.3
}

# Define Paths
SOURCE_DIR = Path("data/dataset-sign-language/asl")
OUTPUT_DIRS = {
    "train": Path("data/asl_alphabet_train"),
    "val": Path("data/asl_alphabet_val"),
    "test": Path("data/asl_alphabet_test")
}

def setup_directories(directories: Dict[str, Path]):
    """Removes existing directories and creates fresh ones."""
    for dir_path in directories.values():
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True)
        print(f"Prepared directory: {dir_path}")

def get_image_paths(class_dir: Path) -> List[Path]:
    """Retrieves all files from a class directory."""
    return [f for f in class_dir.glob("*") if f.is_file()]

def distribute_files(files: List[Path], class_name: str, output_root: Dict[str, Path]):
    """Splits a list of files and copies them to their respective destinations."""
    random.shuffle(files)
    
    total = len(files)
    # Calculate split indices
    train_end = int(total * SPLIT_RATIOS["train"])
    val_end = train_end + int(total * SPLIT_RATIOS["val"])

    # Slice the list into sets
    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    # Copy files to new structure
    for split_name, split_files in splits.items():
        # Create class-specific subdirectory (e.g., train/A/)
        target_dir = output_root[split_name] / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in split_files:
            shutil.copy(file_path, target_dir / file_path.name)

    print(f"✅ Processed '{class_name}': {len(splits['train'])} train, "
          f"{len(splits['val'])} val, {len(splits['test'])} test")

def main():
    """Main execution flow."""
    random.seed(SEED)
    
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} not found.")
        return

    # 1. Prepare target folders
    setup_directories(OUTPUT_DIRS)

    # 2. Iterate through each class (folder) in the source
    class_folders = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    
    if not class_folders:
        print("No class directories found.")
        return

    for class_dir in class_folders:
        images = get_image_paths(class_dir)
        if images:
            distribute_files(images, class_dir.name, OUTPUT_DIRS)

    print("\n🚀 Dataset splitting completed successfully.")

if __name__ == "__main__":
    main()
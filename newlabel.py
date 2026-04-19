

from pathlib import Path
from pellet_analyzer_v2 import export_yolo_dataset, CONFIG

all_images = list(Path(r"C:\Users\sande\Downloads\Pallet Project\Original_images").glob("*.jpeg"))
print(f"Found {len(all_images)} images")

export_yolo_dataset(
    image_paths = all_images,
    output_dir  = r"C:\Users\sande\Downloads\Pallet Project\yolo_dataset",
    cfg         = CONFIG,
    train_split = 0.8
)
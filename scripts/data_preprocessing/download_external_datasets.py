import os
import shutil
from pathlib import Path
import kagglehub

RAW_DATASETS_DIR = Path("data/raw/datasets")

KAGGLE_DATASETS = [
    "fnands/nasa-amazon-lidar-2008-2018",
    "fnands/major-tom-core-s2l1c-ssl4eo-amazonia-embeddings",
    "ceeluna/hydrorivers-dataset",
    "ceeluna/gedi-cluster0-2019-2021"
]

def dataset_name(dataset_id):
    """Возвращает имя датасета из ID"""
    return dataset_id.split("/")[-1]

def download_all_kaggle_datasets():
    RAW_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_id in KAGGLE_DATASETS:
        name = dataset_name(dataset_id) 
        target_dir = RAW_DATASETS_DIR / name
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"⏬ Downloading: {dataset_id}")
        path = kagglehub.dataset_download(dataset_id)

        if path and Path(path).exists():
            for file in Path(path).iterdir():
                shutil.move(str(file), str(target_dir))
            print(f"✅ Saved to: {target_dir}\n")
        else:
            print(f"⚠️ Failed to download or locate {dataset_id}\n")

if __name__ == "__main__":
    download_all_kaggle_datasets()

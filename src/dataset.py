from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent.parent

paths = {
    "rgb": BASE_DIR / "data" / "RGB" / "raw",
    "am4": BASE_DIR / "data" / "AM4" / "raw",
    "at4": BASE_DIR / "data" / "AT4" / "raw",
}

Path(paths["rgb"]).mkdir(parents=True, exist_ok=True)
Path(paths["am4"]).mkdir(parents=True, exist_ok=True)
Path(paths["at4"]).mkdir(parents=True, exist_ok=True)


downloads = {
    "rgb": ["https://doi.org/10.5281/zenodo.3233081", "Amazon Forest Dataset.rar"]
}
#  "am4": ["https://doi.org/10.5281/zenodo.4498086","AMAZON.rar"],
#  "at4": ["https://doi.org/10.5281/zenodo.4498086","ATLANTIC FOREST.rar"]}

for dataset, [url, filename] in downloads.items():
    data_empty = not any(Path(paths[dataset]).iterdir())
    if data_empty:
        print(f"Downloading {dataset} data...")
        path = paths[dataset]
        cmd = ["zenodo_get", "-o", str(path), "-g", filename] + [url]
        subprocess.run(cmd, check=True)
        print(f"Dataset {dataset} downloaded to {str(path)}.")
    else:
        print(
            f"{dataset} data already exists at {str(paths[dataset])}, skipping download."
        )

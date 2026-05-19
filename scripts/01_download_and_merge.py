from pathlib import Path
import subprocess
import zipfile

import pandas as pd
from datasets import load_dataset


KAGGLE_DATASET = "simhadrisadaram/mimic-cxr-dataset"
HF_DATASET_NAME = "erjui/csrrg_findings"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

KAGGLE_DIR = RAW_DIR / "kaggle_mimic"
KAGGLE_ZIP = RAW_DIR / "mimic-cxr-dataset.zip"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)


def download_hf_findings() -> pd.DataFrame:
    print("Loading HuggingFace findings dataset...")
    dataset = load_dataset(HF_DATASET_NAME)
    df = dataset["train"].to_pandas()

    output_path = PROCESSED_DIR / "hf_findings.csv"
    df.to_csv(output_path, index=False)

    print(f"HF rows: {len(df)}")
    print(f"HF columns: {list(df.columns)}")
    print(f"Saved HF findings to {output_path}")

    return df


def download_kaggle_dataset() -> None:
    if any(KAGGLE_DIR.iterdir()):
        print(f"Kaggle dataset already exists in {KAGGLE_DIR}")
        return

    print("Downloading Kaggle dataset...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(RAW_DIR),
        ],
        check=True,
    )

    zip_files = list(RAW_DIR.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError("No Kaggle zip file found after download.")

    zip_path = zip_files[0]
    print(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(KAGGLE_DIR)

    print(f"Extracted Kaggle dataset to {KAGGLE_DIR}")


def inspect_kaggle_files() -> None:
    print("Kaggle files:")
    for path in KAGGLE_DIR.rglob("*"):
        if path.is_file():
            print(path)


def main() -> None:
    download_hf_findings()
    download_kaggle_dataset()
    inspect_kaggle_files()


if __name__ == "__main__":
    main()
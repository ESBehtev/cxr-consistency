from pathlib import Path
import ast

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


RAW_DIR = Path("data/raw/kaggle_mimic")
IMAGE_ROOT = RAW_DIR / "official_data_iccv_final" / "files"

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "mimic_cxr_aug_train.csv"
VALID_CSV = RAW_DIR / "mimic_cxr_aug_validate.csv"


def build_image_index() -> dict:
    print("Building image index...")

    image_index = {}

    image_paths = list(IMAGE_ROOT.rglob("*.jpg"))

    for path in tqdm(image_paths):
        image_index[path.name] = str(path)

    print(f"Indexed {len(image_index)} images")

    return image_index


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text.strip()


def load_tables() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV)
    valid = pd.read_csv(VALID_CSV)

    train["source_split"] = "kaggle_train"
    valid["source_split"] = "kaggle_valid"

    df = pd.concat([train, valid], ignore_index=True)

    print(f"Loaded {len(df)} rows")

    return df


def explode_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("Exploding list-based dataset...")

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            image_list = ast.literal_eval(row["image"])
            view_list = ast.literal_eval(row["view"])
            text_list = ast.literal_eval(row["text"])
        except Exception:
            continue

        for image_path in image_list:
            image_name = Path(image_path).name

            if "lateral" in image_path.lower():
                continue

            if len(view_list) > 0:
                current_view = view_list[0]
            else:
                current_view = "UNKNOWN"

            if current_view not in ["AP", "PA"]:
                continue

            if len(text_list) == 0:
                continue

            report = clean_text(text_list[0])

            if len(report) < 20:
                continue

            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "image_name": image_name,
                    "view": current_view,
                    "report": report,
                    "source_split": row["source_split"],
                }
            )

    exploded_df = pd.DataFrame(rows)

    print(f"Exploded rows: {len(exploded_df)}")

    return exploded_df


def resolve_image_paths(df: pd.DataFrame, image_index: dict) -> pd.DataFrame:
    print("Resolving image paths...")

    df["image_path"] = df["image_name"].map(image_index)

    missing = df["image_path"].isna().sum()

    print(f"Missing image paths: {missing}")

    df = df.dropna(subset=["image_path"]).copy()

    return df


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    patients = df["subject_id"].drop_duplicates()

    train_patients, temp_patients = train_test_split(
        patients,
        test_size=0.2,
        random_state=42,
    )

    valid_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.5,
        random_state=42,
    )

    df["split"] = "none"

    df.loc[df["subject_id"].isin(train_patients), "split"] = "train"
    df.loc[df["subject_id"].isin(valid_patients), "split"] = "valid"
    df.loc[df["subject_id"].isin(test_patients), "split"] = "test"

    return df


def main():
    image_index = build_image_index()

    df = load_tables()

    df = explode_dataset(df)

    df = resolve_image_paths(df, image_index)

    df = df.drop_duplicates(
        subset=["image_path", "report"]
    ).reset_index(drop=True)

    print(f"Final rows: {len(df)}")

    df = make_splits(df)

    output_path = PROCESSED_DIR / "cxr_reports_clean.csv"

    df.to_csv(output_path, index=False)

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    print("\nView distribution:")
    print(df["view"].value_counts())

    print(f"\nSaved dataset to {output_path}")


if __name__ == "__main__":
    main()
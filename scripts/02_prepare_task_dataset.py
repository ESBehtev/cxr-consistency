from pathlib import Path
import argparse
import ast
import re
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


RAW_DIR = Path("data/raw/kaggle_mimic")
IMAGE_ROOT = RAW_DIR / "official_data_iccv_final" / "files"

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = RAW_DIR / "mimic_cxr_aug_train.csv"
VALID_CSV = RAW_DIR / "mimic_cxr_aug_validate.csv"

FRONTAL_VIEWS = {"AP", "PA"}
MIN_SEMANTIC_REPORT_CHARS = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, default=PROCESSED_DIR / "cxr_reports_clean.csv")
    parser.add_argument("--min-report-chars", type=int, default=MIN_SEMANTIC_REPORT_CHARS)
    return parser.parse_args()


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


def strip_report_headers(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\b(findings|impression)\s*:", " ", text, flags=re.IGNORECASE)
    return clean_text(text)


def parse_list(value) -> list:
    if isinstance(value, list):
        return value

    if not isinstance(value, str):
        return []

    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []

    if isinstance(parsed, list):
        return parsed

    return []


def extract_study_id(image_path: str) -> str | None:
    match = re.search(r"/s(\d+)/", str(image_path))
    if match is None:
        return None

    return match.group(1)


def build_view_lookup(row: pd.Series) -> dict[str, str]:
    view_lookup = {}

    for view_name, column in [("AP", "AP"), ("PA", "PA"), ("LATERAL", "Lateral")]:
        for image_path in parse_list(row.get(column, [])):
            view_lookup[str(image_path)] = view_name

    return view_lookup


def build_study_report_map(image_list: list[str], text_list: list[str]) -> dict[str, str] | None:
    study_ids = []

    for image_path in image_list:
        study_id = extract_study_id(image_path)

        if study_id is not None and study_id not in study_ids:
            study_ids.append(study_id)

    if len(study_ids) != len(text_list):
        return None

    return {
        study_id: clean_text(report)
        for study_id, report in zip(study_ids, text_list)
    }


def load_tables() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV)
    valid = pd.read_csv(VALID_CSV)

    train["source_split"] = "kaggle_train"
    valid["source_split"] = "kaggle_valid"

    df = pd.concat([train, valid], ignore_index=True)

    print(f"Loaded {len(df)} rows")

    return df


def explode_dataset(df: pd.DataFrame, min_report_chars: int) -> pd.DataFrame:
    print("Exploding list-based dataset...")

    rows = []
    stats = Counter()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_list = parse_list(row["image"])
        text_list = parse_list(row["text"])

        if not image_list or not text_list:
            stats["missing_image_or_text_list"] += 1
            continue

        study_report_map = build_study_report_map(image_list, text_list)

        if study_report_map is None:
            stats["ambiguous_study_report_mapping"] += 1
            continue

        view_lookup = build_view_lookup(row)

        for image_path in image_list:
            image_name = Path(image_path).name
            study_id = extract_study_id(image_path)

            if study_id is None:
                stats["missing_study_id"] += 1
                continue

            current_view = view_lookup.get(str(image_path), "UNKNOWN")

            if current_view not in FRONTAL_VIEWS:
                stats[f"skipped_view_{current_view}"] += 1
                continue

            report = clean_text(study_report_map.get(study_id, ""))
            semantic_report = strip_report_headers(report)

            if len(semantic_report) < min_report_chars:
                stats["empty_or_short_report"] += 1
                continue

            stats["kept_frontal_image"] += 1

            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "study_id": study_id,
                    "image_name": image_name,
                    "view": current_view,
                    "report": report,
                    "source_split": row["source_split"],
                }
            )

    exploded_df = pd.DataFrame(rows)

    print(f"Exploded rows: {len(exploded_df)}")
    print("Explode stats:")
    for key, value in stats.most_common():
        print(f"  {key}: {value}")

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
    args = parse_args()

    image_index = build_image_index()

    df = load_tables()

    df = explode_dataset(
        df=df,
        min_report_chars=args.min_report_chars,
    )

    df = resolve_image_paths(df, image_index)

    df = df.drop_duplicates(
        subset=["image_path", "report"]
    ).reset_index(drop=True)

    print(f"Final rows: {len(df)}")

    df = make_splits(df)

    output_path = args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nSplit distribution:")
    print(df["split"].value_counts())

    print("\nView distribution:")
    print(df["view"].value_counts())

    print(f"\nSaved dataset to {output_path}")


if __name__ == "__main__":
    main()

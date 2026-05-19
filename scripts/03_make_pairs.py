from pathlib import Path
import random
import re
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


INPUT_CSV = Path("data/processed/cxr_reports_clean.csv")
OUTPUT_DIR = Path("data/pairs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "cxr_consistency_pairs.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

MAX_POSITIVES_PER_SPLIT = None
# Для отладки можно поставить:
# MAX_POSITIVES_PER_SPLIT = {
#     "train": 20000,
#     "valid": 3000,
#     "test": 3000,
# }


PATHOLOGY_KEYWORDS = {
    "pneumothorax": ["pneumothorax"],
    "effusion": ["pleural effusion", "effusion"],
    "edema": ["pulmonary edema", "edema", "vascular congestion"],
    "consolidation": ["consolidation", "airspace opacity", "opacity"],
    "atelectasis": ["atelectasis"],
    "cardiomegaly": ["cardiomegaly", "enlarged cardiac silhouette"],
}

PATHOLOGY_REPLACEMENTS = {
    "pneumothorax": "pleural effusion",
    "pleural effusion": "pneumothorax",
    "effusion": "consolidation",
    "edema": "atelectasis",
    "consolidation": "pulmonary edema",
    "opacity": "pneumothorax",
    "atelectasis": "pleural effusion",
    "cardiomegaly": "pneumothorax",
}

LOCATION_REPLACEMENTS = {
    "left": "right",
    "right": "left",
    "upper": "lower",
    "lower": "upper",
    "apical": "basilar",
    "basilar": "apical",
    "bilateral": "right",
}

SEVERITY_REPLACEMENTS = {
    "mild": "severe",
    "moderate": "mild",
    "severe": "mild",
    "small": "large",
    "large": "small",
    "minimal": "marked",
    "marked": "minimal",
}

NEGATION_PATTERNS = [
    (r"\bno evidence of\b", "evidence of"),
    (r"\bthere is no\b", "there is"),
    (r"\bthere are no\b", "there are"),
    (r"\bwithout\b", "with"),
    (r"\bnegative for\b", "positive for"),
    (r"\babsence of\b", "presence of"),
    (r"\bnot seen\b", "seen"),
    (r"\bnot identified\b", "identified"),
    (r"\bno\b", ""),
]


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).split()).strip()


def detect_pathology_group(text: str) -> str:
    text = str(text).lower()
    labels = []

    for label, keywords in PATHOLOGY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            labels.append(label)

    if not labels:
        return "normal_or_unspecified"

    return "|".join(sorted(labels))


def replace_first_keyword(text: str, replacements: dict) -> str | None:
    lower = text.lower()

    for source, target in replacements.items():
        pattern = r"\b" + re.escape(source) + r"\b"

        if re.search(pattern, lower):
            return re.sub(
                pattern,
                target,
                text,
                count=1,
                flags=re.IGNORECASE,
            )

    return None


def distort_negation(text: str) -> str | None:
    for pattern, replacement in NEGATION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            distorted = re.sub(
                pattern,
                replacement,
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            return normalize_spaces(distorted)

    for pathology in PATHOLOGY_REPLACEMENTS.keys():
        pattern = r"\b" + re.escape(pathology) + r"\b"

        if re.search(pattern, text, flags=re.IGNORECASE):
            distorted = re.sub(
                pattern,
                f"no {pathology}",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
            return normalize_spaces(distorted)

    return None


def distort_pathology(text: str) -> str | None:
    distorted = replace_first_keyword(text, PATHOLOGY_REPLACEMENTS)

    if distorted is None:
        return None

    return normalize_spaces(distorted)


def distort_location(text: str) -> str | None:
    distorted = replace_first_keyword(text, LOCATION_REPLACEMENTS)

    if distorted is None:
        return None

    return normalize_spaces(distorted)


def distort_severity(text: str) -> str | None:
    distorted = replace_first_keyword(text, SEVERITY_REPLACEMENTS)

    if distorted is None:
        return None

    return normalize_spaces(distorted)


def maybe_limit_df(df: pd.DataFrame) -> pd.DataFrame:
    if MAX_POSITIVES_PER_SPLIT is None:
        return df

    limited_parts = []

    for split, max_n in MAX_POSITIVES_PER_SPLIT.items():
        part = df[df["split"] == split].copy()

        if len(part) > max_n:
            part = part.sample(
                n=max_n,
                random_state=RANDOM_SEED,
            )

        limited_parts.append(part)

    return pd.concat(limited_parts, ignore_index=True)


def row_to_positive(row: dict) -> dict:
    return {
        "image_path": row["image_path"],
        "report": row["report"],
        "label": 1,
        "negative_type": "positive",
        "subject_id": row["subject_id"],
        "paired_report_subject_id": row["subject_id"],
        "view": row["view"],
        "split": row["split"],
    }


def make_swapped_negative(row: dict, candidate: dict, negative_type: str) -> dict:
    return {
        "image_path": row["image_path"],
        "report": candidate["report"],
        "label": 0,
        "negative_type": negative_type,
        "subject_id": row["subject_id"],
        "paired_report_subject_id": candidate["subject_id"],
        "view": row["view"],
        "split": row["split"],
    }


def make_distorted_negative(row: dict, distorted_report: str, negative_type: str) -> dict:
    return {
        "image_path": row["image_path"],
        "report": distorted_report,
        "label": 0,
        "negative_type": negative_type,
        "subject_id": row["subject_id"],
        "paired_report_subject_id": row["subject_id"],
        "view": row["view"],
        "split": row["split"],
    }


def build_indices(rows: list[dict]) -> dict:
    indices = {
        "all": rows,
        "by_view": defaultdict(list),
        "by_pathology": defaultdict(list),
    }

    for row in rows:
        indices["by_view"][row["view"]].append(row)
        indices["by_pathology"][row["pathology_group"]].append(row)

    return indices


def sample_different_subject(candidates: list[dict], subject_id: int | str) -> dict | None:
    if not candidates:
        return None

    for _ in range(20):
        candidate = random.choice(candidates)

        if candidate["subject_id"] != subject_id:
            return candidate

    filtered = [
        candidate
        for candidate in candidates
        if candidate["subject_id"] != subject_id
    ]

    if not filtered:
        return None

    return random.choice(filtered)


def make_pairs_for_split(split_df: pd.DataFrame) -> list[dict]:
    pairs = []

    records = split_df.to_dict("records")
    indices = build_indices(records)

    for row in tqdm(records, total=len(records)):
        pairs.append(row_to_positive(row))

        random_candidate = sample_different_subject(
            indices["all"],
            row["subject_id"],
        )

        if random_candidate is not None:
            pairs.append(
                make_swapped_negative(
                    row=row,
                    candidate=random_candidate,
                    negative_type="random_report",
                )
            )

        view_candidate = sample_different_subject(
            indices["by_view"][row["view"]],
            row["subject_id"],
        )

        if view_candidate is None:
            view_candidate = random_candidate

        if view_candidate is not None:
            pairs.append(
                make_swapped_negative(
                    row=row,
                    candidate=view_candidate,
                    negative_type="view_matched_report",
                )
            )

        pathology_candidate = sample_different_subject(
            indices["by_pathology"][row["pathology_group"]],
            row["subject_id"],
        )

        if pathology_candidate is None:
            pathology_candidate = random_candidate

        if pathology_candidate is not None:
            pairs.append(
                make_swapped_negative(
                    row=row,
                    candidate=pathology_candidate,
                    negative_type="pathology_matched_report",
                )
            )

        distorted = distort_negation(row["report"])

        if distorted is not None and distorted != row["report"]:
            pairs.append(
                make_distorted_negative(
                    row=row,
                    distorted_report=distorted,
                    negative_type="distorted_negation",
                )
            )

        distorted = distort_pathology(row["report"])

        if distorted is not None and distorted != row["report"]:
            pairs.append(
                make_distorted_negative(
                    row=row,
                    distorted_report=distorted,
                    negative_type="distorted_pathology",
                )
            )

        distorted = distort_location(row["report"])

        if distorted is not None and distorted != row["report"]:
            pairs.append(
                make_distorted_negative(
                    row=row,
                    distorted_report=distorted,
                    negative_type="distorted_location",
                )
            )

        distorted = distort_severity(row["report"])

        if distorted is not None and distorted != row["report"]:
            pairs.append(
                make_distorted_negative(
                    row=row,
                    distorted_report=distorted,
                    negative_type="distorted_severity",
                )
            )

    return pairs


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    print(f"Loaded clean dataset: {len(df)} rows")

    df["report"] = df["report"].apply(normalize_spaces)
    df["pathology_group"] = df["report"].apply(detect_pathology_group)

    df = maybe_limit_df(df)

    print("\nAfter optional limiting:")
    print(df["split"].value_counts())

    print("\nPathology groups:")
    print(df["pathology_group"].value_counts().head(20))

    all_pairs = []

    for split in ["train", "valid", "test"]:
        split_df = df[df["split"] == split].copy()

        print(f"\nMaking pairs for {split}: {len(split_df)} positives")

        split_pairs = make_pairs_for_split(split_df)

        all_pairs.extend(split_pairs)

    pairs_df = pd.DataFrame(all_pairs)

    pairs_df = pairs_df.drop_duplicates(
        subset=[
            "image_path",
            "report",
            "label",
            "negative_type",
        ]
    )

    pairs_df = pairs_df.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    pairs_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved pairs to {OUTPUT_CSV}")

    print("\nLabels:")
    print(pairs_df["label"].value_counts())

    print("\nPair types:")
    print(pairs_df["negative_type"].value_counts())

    print("\nSplits:")
    print(pairs_df["split"].value_counts())


if __name__ == "__main__":
    main()
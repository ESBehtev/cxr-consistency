from pathlib import Path
import argparse
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
DEFAULT_NEGATIVES_PER_POSITIVE = 2

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--negatives-per-positive",
        type=int,
        default=DEFAULT_NEGATIVES_PER_POSITIVE,
        help="Use -1 to keep all generated negatives.",
    )
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).split()).strip()


def normalize_report_key(text: str) -> str:
    return normalize_spaces(text).lower()


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
        "study_id": row.get("study_id"),
        "paired_report_subject_id": row["subject_id"],
        "paired_report_study_id": row.get("study_id"),
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
        "study_id": row.get("study_id"),
        "paired_report_subject_id": candidate["subject_id"],
        "paired_report_study_id": candidate.get("study_id"),
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
        "study_id": row.get("study_id"),
        "paired_report_subject_id": row["subject_id"],
        "paired_report_study_id": row.get("study_id"),
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


def add_if_unique_negative(negatives: list[dict], candidate: dict, seen_reports: set[str]) -> None:
    report_key = normalize_report_key(candidate["report"])

    if report_key in seen_reports:
        return

    seen_reports.add(report_key)
    negatives.append(candidate)


def make_pairs_for_split(
    split_df: pd.DataFrame,
    negatives_per_positive: int | None,
) -> list[dict]:
    pairs = []

    records = split_df.to_dict("records")
    indices = build_indices(records)

    for row in tqdm(records, total=len(records)):
        pairs.append(row_to_positive(row))
        row_report_key = normalize_report_key(row["report"])
        row_negatives = []
        seen_negative_reports = {row_report_key}

        random_candidate = sample_different_subject(
            indices["all"],
            row["subject_id"],
        )

        if random_candidate is not None:
            add_if_unique_negative(
                row_negatives,
                make_swapped_negative(
                    row=row,
                    candidate=random_candidate,
                    negative_type="random_report",
                ),
                seen_negative_reports,
            )

        view_candidate = sample_different_subject(
            indices["by_view"][row["view"]],
            row["subject_id"],
        )

        if view_candidate is not None:
            add_if_unique_negative(
                row_negatives,
                make_swapped_negative(
                    row=row,
                    candidate=view_candidate,
                    negative_type="view_matched_report",
                ),
                seen_negative_reports,
            )

        pathology_candidate = None
        pathology_group = row["pathology_group"]

        if pathology_group != "normal_or_unspecified":
            pathology_candidate = sample_different_subject(
                indices["by_pathology"][pathology_group],
                row["subject_id"],
            )

        if pathology_candidate is not None:
            add_if_unique_negative(
                row_negatives,
                make_swapped_negative(
                    row=row,
                    candidate=pathology_candidate,
                    negative_type="pathology_matched_report",
                ),
                seen_negative_reports,
            )

        distorted = distort_negation(row["report"])

        if distorted is not None and distorted != row["report"]:
            add_if_unique_negative(
                row_negatives,
                make_distorted_negative(
                    row=row,
                    distorted_report=distorted,
                    negative_type="distorted_negation",
                ),
                seen_negative_reports,
            )

        distorted = distort_pathology(row["report"])

        if distorted is not None and distorted != row["report"]:
            add_if_unique_negative(
                row_negatives,
                make_distorted_negative(
                    row=row,
                    distorted_report=distorted,
                    negative_type="distorted_pathology",
                ),
                seen_negative_reports,
            )

        distorted = distort_location(row["report"])

        if distorted is not None and distorted != row["report"]:
            add_if_unique_negative(
                row_negatives,
                make_distorted_negative(
                    row=row,
                    distorted_report=distorted,
                    negative_type="distorted_location",
                ),
                seen_negative_reports,
            )

        distorted = distort_severity(row["report"])

        if distorted is not None and distorted != row["report"]:
            add_if_unique_negative(
                row_negatives,
                make_distorted_negative(
                    row=row,
                    distorted_report=distorted,
                    negative_type="distorted_severity",
                ),
                seen_negative_reports,
            )

        if negatives_per_positive is not None and len(row_negatives) > negatives_per_positive:
            row_negatives = random.sample(row_negatives, k=negatives_per_positive)

        pairs.extend(row_negatives)

    return pairs


def remove_contradictory_negatives(pairs_df: pd.DataFrame) -> pd.DataFrame:
    pairs_df = pairs_df.copy()
    pairs_df["report_key"] = pairs_df["report"].apply(normalize_report_key)

    positive_keys = set(
        pairs_df.loc[pairs_df["label"] == 1, ["image_path", "report_key"]]
        .itertuples(index=False, name=None)
    )

    pair_keys = list(
        pairs_df[["image_path", "report_key"]].itertuples(index=False, name=None)
    )
    contradictory_mask = (
        pairs_df["label"].eq(0)
        & pd.Series(
            [key in positive_keys for key in pair_keys],
            index=pairs_df.index,
        )
    )

    removed = int(contradictory_mask.sum())
    if removed:
        print(f"\nRemoved contradictory negative pairs: {removed}")

    pairs_df = pairs_df.loc[~contradictory_mask].copy()

    before = len(pairs_df)
    pairs_df = pairs_df.drop_duplicates(
        subset=[
            "image_path",
            "report_key",
            "label",
        ]
    ).copy()
    removed_duplicates = before - len(pairs_df)

    if removed_duplicates:
        print(f"Removed duplicate pairs: {removed_duplicates}")

    return pairs_df.drop(columns=["report_key"])


def main() -> None:
    args = parse_args()
    negatives_per_positive = (
        None
        if args.negatives_per_positive < 0
        else args.negatives_per_positive
    )

    df = pd.read_csv(args.input_csv)

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

        split_pairs = make_pairs_for_split(
            split_df=split_df,
            negatives_per_positive=negatives_per_positive,
        )

        all_pairs.extend(split_pairs)

    pairs_df = pd.DataFrame(all_pairs)

    pairs_df = remove_contradictory_negatives(pairs_df)

    pairs_df = pairs_df.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(args.output_csv, index=False)

    print(f"\nSaved pairs to {args.output_csv}")

    print("\nLabels:")
    print(pairs_df["label"].value_counts())

    print("\nPair types:")
    print(pairs_df["negative_type"].value_counts())

    print("\nSplits:")
    print(pairs_df["split"].value_counts())


if __name__ == "__main__":
    main()

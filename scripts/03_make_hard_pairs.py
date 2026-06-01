from pathlib import Path
import argparse
import random
import re
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

INPUT_CSV = Path("data/processed/cxr_reports_clean.csv")
OUTPUT_CSV = Path("data/pairs/cxr_consistency_pairs_hard.csv")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PATHOLOGY_KEYWORDS = {
    "pneumothorax": ["pneumothorax"],
    "effusion": ["pleural effusion", "effusion"],
    "edema": ["pulmonary edema", "edema", "vascular congestion"],
    "consolidation": ["consolidation", "airspace opacity", "opacity", "infiltrate"],
    "atelectasis": ["atelectasis"],
    "cardiomegaly": ["cardiomegaly", "enlarged cardiac silhouette"],
}

PATHOLOGY_TERMS = [
    "pleural effusion", "effusion", "pneumothorax", "edema", "pulmonary edema",
    "vascular congestion", "consolidation", "airspace opacity", "opacity",
    "infiltrate", "infiltration", "atelectasis", "cardiomegaly", "nodule",
    "nodular", "mass", "lesion", "airspace disease", "interstitial", "infection",
]

DEVICE_TERMS = [
    "tube", "tubes", "catheter", "catheters", "line", "lines", "picc", "port",
    "lead", "leads", "wire", "wires", "device", "devices", "pacemaker",
    "clip", "clips", "sternotomy", "endotracheal", "enteric", "ng tube",
    "feeding tube", "central venous", "tip", "terminates", "projects over",
    "chest tube", "support apparatus", "hardware", "suture", "sutures",
    "surgical", "postsurgical", "postoperative", "post-operative",
    "placement", "position", "removed", "removal",
]

ADMIN_TERMS = [
    "posted", "communication", "communicated", "telephone", "discussed",
    "provider", "radiology", "critical", "final report", "preliminary",
    "wet read", "comparison", "portable chest",
]

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

TEMPORAL_REPLACEMENTS = [
    (r"\bimproved\b", "worsened"),
    (r"\bimprovement\b", "worsening"),
    (r"\bworsened\b", "improved"),
    (r"\bworsening\b", "improvement"),
    (r"\bincreasing\b", "decreasing"),
    (r"\bincreased\b", "decreased"),
    (r"\bdecreasing\b", "increasing"),
    (r"\bdecreased\b", "increased"),
    (r"\bstable\b", "progressed"),
    (r"\bunchanged\b", "progressed"),
    (r"\bprogressed\b", "stable"),
    (r"\bprogression\b", "stability"),
    (r"\bresolved\b", "persistent"),
    (r"\bresolution\b", "persistence"),
    (r"\bpersistent\b", "resolved"),
    (r"\bpersistence\b", "resolution"),
]

SEMANTIC_SWAPS = [
    ("pulmonary edema", "consolidation"),
    ("edema", "consolidation"),
    ("consolidation", "edema"),
    ("atelectasis", "consolidation"),
    ("opacity", "infiltrate"),
    ("opacities", "infiltrates"),
    ("infiltrate", "opacity"),
    ("infiltrates", "opacities"),
]

PARTIAL_SEVERITY_SWAPS = [
    (r"\bmild\b", "severe"),
    (r"\bmoderate\b", "severe"),
    (r"\bsevere\b", "mild"),
    (r"\bsmall\b", "large"),
    (r"\blarge\b", "small"),
    (r"\btrace\b", "moderate"),
    (r"\bminimal\b", "moderate"),
]

DUPLICATED_PATHOLOGY_RE = re.compile(
    r"\b(edema|consolidation|atelectasis|opacity|opacities|infiltrate|"
    r"infiltrates|effusion|pneumothorax)\b(?:\W+\1\b)+",
    flags=re.IGNORECASE,
)

TEMPORAL_CONTEXT_RE = re.compile(
    r"\b(improved|improvement|improving|worsened|worsening|increasing|"
    r"increased|decreasing|decreased|stable|unchanged|progressed|"
    r"progression|resolved|resolution|persistent|persistence)\b",
    flags=re.IGNORECASE,
)

NEGATED_CONTEXT_RE = re.compile(
    r"\b(no|without|negative for|absence of|not seen|not identified)\b",
    flags=re.IGNORECASE,
)

@dataclass(frozen=True)
class Distortion:
    text: str
    kind: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--max-hard-per-positive", type=int, default=4)
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).split()).strip()


def normalize_report_key(text: str) -> str:
    return normalize_spaces(text).lower()


def split_sentences(text: str) -> list[str]:
    text = normalize_spaces(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def replace_sentence(text: str, old: str, new: str) -> str:
    idx = text.find(old)
    if idx < 0:
        return text
    return normalize_spaces(text[:idx] + new + text[idx + len(old):])


def has_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(re.search(r"\b" + re.escape(term) + r"\b", lower) for term in terms)


def has_pathology(sentence: str) -> bool:
    return has_any(sentence, PATHOLOGY_TERMS)


def has_device_context(sentence: str) -> bool:
    return has_any(sentence, DEVICE_TERMS)


def has_admin_context(sentence: str) -> bool:
    return has_any(sentence, ADMIN_TERMS)


def eligible_pathology_sentence(sentence: str) -> bool:
    return has_pathology(sentence) and not has_device_context(sentence) and not has_admin_context(sentence)


def has_duplicated_pathology(sentence: str) -> bool:
    return bool(DUPLICATED_PATHOLOGY_RE.search(sentence))


def detect_pathology_group(text: str) -> str:
    lower = str(text).lower()
    labels = []
    for label, keywords in PATHOLOGY_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            labels.append(label)
    return "|".join(sorted(labels)) if labels else "normal_or_unspecified"


def sample_different_subject(candidates: list[dict], subject_id: int | str) -> dict | None:
    if not candidates:
        return None
    for _ in range(30):
        candidate = random.choice(candidates)
        if candidate["subject_id"] != subject_id:
            return candidate
    filtered = [c for c in candidates if c["subject_id"] != subject_id]
    return random.choice(filtered) if filtered else None


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


def distort_negation(text: str) -> str | None:
    for pattern, replacement in NEGATION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return normalize_spaces(re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE))
    return None


def laterality_conflict(text: str) -> str | None:
    for sentence in split_sentences(text):
        if not eligible_pathology_sentence(sentence) or has_duplicated_pathology(sentence):
            continue
        if not re.search(r"\b(left|right)\b", sentence, flags=re.IGNORECASE):
            continue

        def swap(m):
            word = m.group(0)
            repl = "right" if word.lower() == "left" else "left"
            return repl.capitalize() if word[0].isupper() else repl

        new_sentence = re.sub(r"\b(left|right)\b", swap, sentence, count=1, flags=re.IGNORECASE)
        if new_sentence != sentence:
            return replace_sentence(text, sentence, new_sentence)
    return None


def temporal_mismatch(text: str) -> str | None:
    for sentence in split_sentences(text):
        if not eligible_pathology_sentence(sentence) or has_duplicated_pathology(sentence):
            continue
        for pattern, replacement in TEMPORAL_REPLACEMENTS:
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                new_sentence = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
                if new_sentence != sentence:
                    return replace_sentence(text, sentence, new_sentence)
    return None


def pathology_semantic_swap(text: str) -> str | None:
    for sentence in split_sentences(text):
        if not eligible_pathology_sentence(sentence) or has_duplicated_pathology(sentence):
            continue
        if len(re.findall(r"\b[a-zA-Z]{2,}\b", sentence)) < 5:
            continue
        if TEMPORAL_CONTEXT_RE.search(sentence) or NEGATED_CONTEXT_RE.search(sentence):
            continue
        lower_sentence = sentence.lower()
        for source, target in SEMANTIC_SWAPS:
            source_pattern = r"\b" + re.escape(source) + r"\b"
            target_pattern = r"\b" + re.escape(target) + r"\b"
            if not re.search(source_pattern, lower_sentence):
                continue
            if re.search(target_pattern, lower_sentence):
                continue
            new_sentence = re.sub(source_pattern, target, sentence, count=1, flags=re.IGNORECASE)
            if new_sentence != sentence and not has_duplicated_pathology(new_sentence):
                return replace_sentence(text, sentence, new_sentence)
    return None


def partial_mismatch(text: str) -> str | None:
    sentences = split_sentences(text)
    pathology_sentences = [
        s for s in sentences
        if eligible_pathology_sentence(s) and not has_duplicated_pathology(s)
    ]
    if len(pathology_sentences) < 2:
        return None
    for sentence in pathology_sentences:
        for pattern, replacement in PARTIAL_SEVERITY_SWAPS:
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                new_sentence = re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
                if new_sentence != sentence:
                    return replace_sentence(text, sentence, new_sentence)
    return None


def make_hard_distortions(text: str) -> list[Distortion]:
    funcs = [
        ("distorted_negation", distort_negation),
        ("laterality_conflict", laterality_conflict),
        ("temporal_mismatch", temporal_mismatch),
        ("pathology_semantic_swap", pathology_semantic_swap),
        ("partial_mismatch", partial_mismatch),
    ]
    out = []
    original_key = normalize_report_key(text)
    seen = {original_key}
    for kind, func in funcs:
        distorted = func(text)
        if distorted is None:
            continue
        distorted = normalize_spaces(distorted)
        key = normalize_report_key(distorted)
        if key in seen:
            continue
        seen.add(key)
        out.append(Distortion(distorted, kind))
    return out


def build_indices(records: list[dict]) -> dict:
    by_pathology = defaultdict(list)
    for row in records:
        by_pathology[row["pathology_group"]].append(row)
    return {"all": records, "by_pathology": by_pathology}


def add_if_unique_negative(negatives: list[dict], candidate: dict, seen_reports: set[str]) -> None:
    key = normalize_report_key(candidate["report"])
    if key in seen_reports:
        return
    seen_reports.add(key)
    negatives.append(candidate)


def make_pairs_for_split(split_df: pd.DataFrame, max_hard_per_positive: int) -> list[dict]:
    pairs = []
    records = split_df.to_dict("records")
    indices = build_indices(records)

    for row in tqdm(records, total=len(records)):
        pairs.append(row_to_positive(row))
        seen_negative_reports = {normalize_report_key(row["report"])}
        row_negatives = []

        random_candidate = sample_different_subject(indices["all"], row["subject_id"])
        if random_candidate is not None:
            add_if_unique_negative(
                row_negatives,
                make_swapped_negative(row, random_candidate, "random_report"),
                seen_negative_reports,
            )

        pathology_group = row["pathology_group"]
        if pathology_group != "normal_or_unspecified":
            pathology_candidate = sample_different_subject(
                indices["by_pathology"][pathology_group], row["subject_id"]
            )
            if pathology_candidate is not None:
                add_if_unique_negative(
                    row_negatives,
                    make_swapped_negative(row, pathology_candidate, "pathology_matched_report"),
                    seen_negative_reports,
                )

        hard = make_hard_distortions(row["report"])
        if max_hard_per_positive >= 0 and len(hard) > max_hard_per_positive:
            hard = random.sample(hard, max_hard_per_positive)
        for distortion in hard:
            add_if_unique_negative(
                row_negatives,
                make_distorted_negative(row, distortion.text, distortion.kind),
                seen_negative_reports,
            )

        pairs.extend(row_negatives)
    return pairs


def remove_contradictory_negatives(pairs_df: pd.DataFrame) -> pd.DataFrame:
    pairs_df = pairs_df.copy()
    pairs_df["report_key"] = pairs_df["report"].apply(normalize_report_key)
    positive_keys = set(
        pairs_df.loc[pairs_df["label"] == 1, ["image_path", "report_key"]]
        .itertuples(index=False, name=None)
    )
    pair_keys = list(pairs_df[["image_path", "report_key"]].itertuples(index=False, name=None))
    contradictory_mask = pairs_df["label"].eq(0) & pd.Series(
        [key in positive_keys for key in pair_keys], index=pairs_df.index
    )
    removed = int(contradictory_mask.sum())
    if removed:
        print(f"Removed contradictory negatives: {removed}")
    pairs_df = pairs_df.loc[~contradictory_mask].copy()
    before = len(pairs_df)
    pairs_df = pairs_df.drop_duplicates(subset=["image_path", "report_key", "label"]).copy()
    if before - len(pairs_df):
        print(f"Removed duplicate pairs: {before - len(pairs_df)}")
    return pairs_df.drop(columns=["report_key"])


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    print(f"Loaded clean dataset: {len(df)} rows")
    df["report"] = df["report"].apply(normalize_spaces)
    df["pathology_group"] = df["report"].apply(detect_pathology_group)

    all_pairs = []
    for split in ["train", "valid", "test"]:
        split_df = df[df["split"] == split].copy()
        print(f"Making hard pairs for {split}: {len(split_df)} positives")
        all_pairs.extend(make_pairs_for_split(split_df, args.max_hard_per_positive))

    pairs_df = pd.DataFrame(all_pairs)
    pairs_df = remove_contradictory_negatives(pairs_df)
    pairs_df = pairs_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(args.output_csv, index=False)

    print(f"Saved pairs to {args.output_csv}")
    print("Labels:")
    print(pairs_df["label"].value_counts())
    print("Pair types:")
    print(pairs_df["negative_type"].value_counts())
    print("Splits:")
    print(pairs_df["split"].value_counts())

if __name__ == "__main__":
    main()

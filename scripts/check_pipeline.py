from pathlib import Path
import argparse
import re
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))


DEFAULT_CLEAN_CSV = Path("data/processed/cxr_reports_clean.csv")
DEFAULT_PAIRS_CSV = Path("data/pairs/cxr_consistency_pairs.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-csv", type=Path, default=DEFAULT_CLEAN_CSV)
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--tokenizer-sample-size", type=int, default=10000)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).split()).strip()


def report_key(text: str) -> str:
    return normalize_spaces(text).lower()


def semantic_report_text(text: str) -> str:
    text = normalize_spaces(text)
    text = re.sub(r"\b(findings|impression)\s*:", " ", text, flags=re.IGNORECASE)
    return normalize_spaces(text)


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def check_split_leakage(df: pd.DataFrame, id_col: str, label: str) -> int:
    errors = 0
    splits = sorted(df["split"].dropna().unique())

    for idx, split_a in enumerate(splits):
        values_a = set(df.loc[df["split"] == split_a, id_col].dropna())

        for split_b in splits[idx + 1:]:
            values_b = set(df.loc[df["split"] == split_b, id_col].dropna())
            overlap = len(values_a & values_b)
            print(f"{label} overlap {split_a}/{split_b}: {overlap}")

            if overlap:
                errors += 1

    return errors


def check_clean_dataset(clean_df: pd.DataFrame) -> int:
    errors = 0

    print_section("Clean Dataset")
    print(f"rows: {len(clean_df)}")
    print("splits:")
    print(clean_df["split"].value_counts(dropna=False).to_string())

    if "subject_id" not in clean_df.columns:
        print("ERROR: missing subject_id")
        errors += 1
    else:
        errors += check_split_leakage(clean_df, "subject_id", "patient")

    if "image_path" not in clean_df.columns:
        print("ERROR: missing image_path")
        errors += 1
    else:
        errors += check_split_leakage(clean_df, "image_path", "image")
        broken = int((~clean_df["image_path"].map(lambda p: Path(str(p)).exists())).sum())
        print(f"broken image paths: {broken}")
        if broken:
            errors += 1

    if "report" not in clean_df.columns:
        print("ERROR: missing report")
        errors += 1
    else:
        semantic = clean_df["report"].fillna("").map(semantic_report_text)
        empty_reports = int(semantic.eq("").sum())
        short_reports = int(semantic.str.len().lt(20).sum())
        print(f"empty reports after header strip: {empty_reports}")
        print(f"short reports after header strip (<20 chars): {short_reports}")
        if empty_reports or short_reports:
            errors += 1

        clean_df = clean_df.copy()
        clean_df["report_key"] = clean_df["report"].map(report_key)
        report_overlap_errors = 0
        splits = sorted(clean_df["split"].dropna().unique())
        for idx, split_a in enumerate(splits):
            reports_a = set(clean_df.loc[clean_df["split"] == split_a, "report_key"])
            for split_b in splits[idx + 1:]:
                reports_b = set(clean_df.loc[clean_df["split"] == split_b, "report_key"])
                overlap = len(reports_a & reports_b)
                print(f"report text overlap {split_a}/{split_b}: {overlap}")
                if overlap:
                    report_overlap_errors += 1
        if report_overlap_errors:
            print("NOTE: report text overlap is not patient leakage, but can create text shortcuts.")

    if {"image_path", "report"}.issubset(clean_df.columns):
        duplicates = int(clean_df.duplicated(subset=["image_path", "report"]).sum())
        print(f"duplicate clean image/report rows: {duplicates}")
        if duplicates:
            errors += 1

    return errors


def check_pairs(pairs_df: pd.DataFrame) -> int:
    errors = 0

    print_section("Pairs")
    print(f"rows: {len(pairs_df)}")
    print("splits:")
    print(pairs_df["split"].value_counts(dropna=False).to_string())
    print("labels:")
    print(pairs_df["label"].value_counts(dropna=False).to_string())
    print("negative types:")
    print(pairs_df["negative_type"].value_counts(dropna=False).to_string())
    print("labels by split:")
    print(pd.crosstab(pairs_df["split"], pairs_df["label"]).to_string())

    if "subject_id" in pairs_df.columns:
        errors += check_split_leakage(pairs_df, "subject_id", "pair patient")

    if "image_path" in pairs_df.columns:
        errors += check_split_leakage(pairs_df, "image_path", "pair image")

    pairs_df = pairs_df.copy()
    pairs_df["report_key"] = pairs_df["report"].map(report_key)

    contradictory = int(
        (pairs_df.groupby(["image_path", "report_key"])["label"].nunique() > 1).sum()
    )
    duplicate_pairs = int(
        pairs_df.duplicated(subset=["image_path", "report_key", "label"]).sum()
    )

    print(f"contradictory image/report labels: {contradictory}")
    print(f"duplicate image/report/label pairs: {duplicate_pairs}")

    if contradictory:
        errors += 1

    if duplicate_pairs:
        errors += 1

    if {"label", "negative_type", "subject_id", "paired_report_subject_id"}.issubset(pairs_df.columns):
        same_subject_swaps = int(
            (
                (pairs_df["label"] == 0)
                & pairs_df["negative_type"].isin(
                    ["random_report", "view_matched_report", "pathology_matched_report"]
                )
                & (pairs_df["subject_id"] == pairs_df["paired_report_subject_id"])
            ).sum()
        )
        print(f"swapped negatives with same subject: {same_subject_swaps}")
        if same_subject_swaps:
            errors += 1

    return errors


def check_tokenizer_truncation(
    pairs_df: pd.DataFrame,
    tokenizer_name: str,
    max_length: int,
    sample_size: int,
    trust_remote_code: bool,
    local_files_only: bool,
) -> None:
    print_section("Tokenizer")
    if tokenizer_name == "simple":
        from cxr_consistency.tokenizer import SimpleHashTokenizer

        tokenizer = SimpleHashTokenizer()
    else:
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            print(f"WARNING: tokenizer check skipped: {exc}")
            return

    reports = pairs_df["report"].dropna()
    if len(reports) > sample_size:
        reports = reports.sample(n=sample_size, random_state=42)

    lengths = reports.map(
        lambda text: len(
            tokenizer(
                str(text),
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
        )
    )

    print(lengths.quantile([0.0, 0.5, 0.9, 0.95, 0.99, 1.0]).to_string())
    print(f"fraction > max_length={max_length}: {(lengths > max_length).mean():.4f}")


def main() -> None:
    args = parse_args()

    clean_df = pd.read_csv(args.clean_csv)
    pairs_df = pd.read_csv(args.pairs_csv)

    errors = 0
    errors += check_clean_dataset(clean_df)
    errors += check_pairs(pairs_df)

    if args.tokenizer_name is not None:
        check_tokenizer_truncation(
            pairs_df=pairs_df,
            tokenizer_name=args.tokenizer_name,
            max_length=args.max_length,
            sample_size=args.tokenizer_sample_size,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )

    print_section("Result")
    if errors:
        raise SystemExit(f"FAILED sanity checks: {errors}")

    print("OK")


if __name__ == "__main__":
    main()

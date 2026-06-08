from __future__ import annotations

import difflib
import random
import re
from dataclasses import dataclass

import pandas as pd


PATHOLOGY_TERMS = [
    "pleural effusion", "effusion", "pneumothorax", "edema", "pulmonary edema",
    "vascular congestion", "consolidation", "airspace opacity", "opacity",
    "infiltrate", "infiltration", "atelectasis", "cardiomegaly", "nodule",
    "nodular", "mass", "lesion", "airspace disease", "interstitial", "infection",
]
DEVICE_TERMS = [
    "tube", "catheter", "line", "picc", "port", "lead", "wire", "device",
    "pacemaker", "endotracheal", "enteric", "central venous", "tip",
    "chest tube", "support apparatus", "hardware", "surgical", "postoperative",
]
ADMIN_TERMS = ["posted", "communication", "telephone", "provider", "radiology", "comparison"]
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
    (r"\bimproved\b", "worsened"), (r"\bimprovement\b", "worsening"),
    (r"\bworsened\b", "improved"), (r"\bworsening\b", "improvement"),
    (r"\bincreasing\b", "decreasing"), (r"\bincreased\b", "decreased"),
    (r"\bdecreasing\b", "increasing"), (r"\bdecreased\b", "increased"),
    (r"\bstable\b", "progressed"), (r"\bunchanged\b", "progressed"),
    (r"\bprogressed\b", "stable"), (r"\bresolved\b", "persistent"),
    (r"\bpersistent\b", "resolved"),
]
SEMANTIC_SWAPS = [
    ("pulmonary edema", "consolidation"), ("edema", "consolidation"),
    ("consolidation", "edema"), ("atelectasis", "consolidation"),
    ("opacity", "infiltrate"), ("opacities", "infiltrates"),
    ("infiltrate", "opacity"), ("infiltrates", "opacities"),
    ("pleural effusion", "pneumothorax"), ("pneumothorax", "pleural effusion"),
]
PARTIAL_SEVERITY_SWAPS = [
    (r"\bmild\b", "severe"), (r"\bmoderate\b", "severe"),
    (r"\bsevere\b", "mild"), (r"\bsmall\b", "large"),
    (r"\blarge\b", "small"), (r"\btrace\b", "moderate"),
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
class NegativePair:
    report: str
    kind: str
    diff_html: str


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).split()).strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_spaces(text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def replace_sentence(text: str, old: str, new: str) -> str:
    idx = text.find(old)
    if idx < 0:
        return text
    return normalize_spaces(text[:idx] + new + text[idx + len(old):])


def has_any(text: str, terms: list[str]) -> bool:
    lower = text.lower()
    return any(re.search(r"\b" + re.escape(term) + r"\b", lower) for term in terms)


def eligible_pathology_sentence(sentence: str) -> bool:
    return (
        has_any(sentence, PATHOLOGY_TERMS)
        and not has_any(sentence, DEVICE_TERMS)
        and not has_any(sentence, ADMIN_TERMS)
        and not DUPLICATED_PATHOLOGY_RE.search(sentence)
    )


def distort_negation(text: str) -> str | None:
    for pattern, replacement in NEGATION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return normalize_spaces(re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE))
    return None


def laterality_conflict(text: str) -> str | None:
    for sentence in split_sentences(text):
        if not eligible_pathology_sentence(sentence):
            continue
        if not re.search(r"\b(left|right)\b", sentence, flags=re.IGNORECASE):
            continue

        def swap(match: re.Match) -> str:
            word = match.group(0)
            repl = "right" if word.lower() == "left" else "left"
            return repl.capitalize() if word[0].isupper() else repl

        new_sentence = re.sub(r"\b(left|right)\b", swap, sentence, count=1, flags=re.IGNORECASE)
        if new_sentence != sentence:
            return replace_sentence(text, sentence, new_sentence)
    return None


def temporal_mismatch(text: str) -> str | None:
    for sentence in split_sentences(text):
        if not eligible_pathology_sentence(sentence):
            continue
        for pattern, replacement in TEMPORAL_REPLACEMENTS:
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                return replace_sentence(text, sentence, re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE))
    return None


def pathology_semantic_swap(text: str) -> str | None:
    for sentence in split_sentences(text):
        if not eligible_pathology_sentence(sentence):
            continue
        if TEMPORAL_CONTEXT_RE.search(sentence) or NEGATED_CONTEXT_RE.search(sentence):
            continue
        for source, target in SEMANTIC_SWAPS:
            pattern = r"\b" + re.escape(source) + r"\b"
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                return replace_sentence(text, sentence, re.sub(pattern, target, sentence, count=1, flags=re.IGNORECASE))
    return None


def partial_mismatch(text: str) -> str | None:
    for sentence in split_sentences(text):
        if not eligible_pathology_sentence(sentence):
            continue
        for pattern, replacement in PARTIAL_SEVERITY_SWAPS:
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                return replace_sentence(text, sentence, re.sub(pattern, replacement, sentence, count=1, flags=re.IGNORECASE))
    return None


def random_report(text: str, positives_df: pd.DataFrame, subject_id: str) -> str | None:
    candidates = positives_df[positives_df["subject_id"].astype(str) != str(subject_id)]
    if candidates.empty:
        candidates = positives_df
    if candidates.empty:
        return None
    sample = candidates.sample(n=1)
    report = normalize_spaces(sample.iloc[0]["report"])
    return report if report and report != normalize_spaces(text) else None


def fallback_negative(text: str, kind: str) -> str:
    text = normalize_spaces(text)
    suffixes = {
        "distorted_negation": " There is a new pneumothorax.",
        "laterality_conflict": " The main abnormality is on the opposite side.",
        "temporal_mismatch": " Compared with prior imaging, the finding has worsened rather than improved.",
        "partial_mismatch": " A previously mild abnormality is now described as severe.",
        "pathology_semantic_swap": " The dominant finding is changed to consolidation.",
    }
    return normalize_spaces(text + suffixes.get(kind, " The report now describes a conflicting finding."))


def html_diff(original: str, modified: str) -> str:
    differ = difflib.HtmlDiff(wrapcolumn=80)
    return differ.make_table(
        original.split(),
        modified.split(),
        fromdesc="Original",
        todesc="Modified",
        context=True,
        numlines=2,
    )


def make_negative(kind: str, original_report: str, positives_df: pd.DataFrame, subject_id: str) -> NegativePair:
    mapping = {
        "distorted_negation": distort_negation,
        "laterality_conflict": laterality_conflict,
        "temporal_mismatch": temporal_mismatch,
        "partial_mismatch": partial_mismatch,
        "pathology_semantic_swap": pathology_semantic_swap,
    }
    if kind == "random_report":
        modified = random_report(original_report, positives_df, subject_id)
    else:
        modified = mapping[kind](original_report)
    if not modified:
        modified = fallback_negative(original_report, kind)
    return NegativePair(report=modified, kind=kind, diff_html=html_diff(original_report, modified))


BUTTONS = [
    ("Negation", "distorted_negation"),
    ("Laterality", "laterality_conflict"),
    ("Temporal", "temporal_mismatch"),
    ("Partial Mismatch", "partial_mismatch"),
    ("Pathology Swap", "pathology_semantic_swap"),
    ("Random Report", "random_report"),
]

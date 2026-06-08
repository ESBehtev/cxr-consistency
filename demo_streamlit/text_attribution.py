from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from inference import load_demo_model, prepare_inputs, predict_probability


TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9'-]*\b")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "there", "this", "to", "was", "were", "with", "without", "within",
    "findings", "impression", "comparison", "portable", "chest", "xray", "x-ray",
    "ap", "pa", "lateral", "view", "image", "exam", "radiograph", "patient",
    "again", "also", "now", "new", "prior", "previous", "unchanged",
}


@dataclass(frozen=True)
class TokenCandidate:
    word: str
    start: int
    end: int


def _token_candidates(report: str, max_words: int) -> list[TokenCandidate]:
    candidates: list[TokenCandidate] = []
    scanned_words = 0
    for match in TOKEN_RE.finditer(report):
        scanned_words += 1
        if scanned_words > max_words:
            break
        word = match.group(0)
        normalized = word.lower().strip("'-")
        if len(normalized) < 3:
            continue
        if normalized in STOP_WORDS:
            continue
        if normalized.isdigit():
            continue
        candidates.append(TokenCandidate(word=word, start=match.start(), end=match.end()))
    return candidates


def _mask_word(report: str, candidate: TokenCandidate, mask_token: str) -> str:
    return report[:candidate.start] + mask_token + report[candidate.end:]


def _normalize_score(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return min(max(value / max_value, 0.0), 1.0)


def highlighted_report(report: str, attributions: pd.DataFrame, max_words: int = 120) -> str:
    if attributions.empty:
        return f"<div>{html.escape(report)}</div>"

    scores = {
        (int(row.start), int(row.end)): float(row.importance)
        for row in attributions.itertuples(index=False)
    }
    max_score = max(scores.values()) if scores else 0.0
    pieces: list[str] = []
    cursor = 0
    analyzed_end = 0

    for candidate in _token_candidates(report, max_words=max_words):
        analyzed_end = max(analyzed_end, candidate.end)
        key = (candidate.start, candidate.end)
        if key not in scores:
            continue
        pieces.append(html.escape(report[cursor:candidate.start]))
        intensity = _normalize_score(scores[key], max_score)
        alpha = 0.18 + intensity * 0.58
        border = 0.35 + intensity * 0.65
        pieces.append(
            "<span style=\""
            f"background: rgba(250, 204, 21, {alpha:.3f}); "
            f"border-bottom: 2px solid rgba(180, 83, 9, {border:.3f}); "
            "padding: 0 2px; border-radius: 3px;\">"
            f"{html.escape(report[candidate.start:candidate.end])}"
            "</span>"
        )
        cursor = candidate.end

    pieces.append(html.escape(report[cursor:]))
    suffix = ""
    if analyzed_end and analyzed_end < len(report):
        suffix = "<div style=\"color:#64748b;font-size:0.82rem;margin-top:8px;\">Analyzed first words only for speed.</div>"
    return f"<div style=\"line-height:1.5;font-size:0.94rem;\">{''.join(pieces)}</div>{suffix}"


def compute_text_attribution(
    image_path: Path,
    report: str,
    original_probability: float | None = None,
    max_words: int = 100,
    top_k: int = 10,
    mask_token: str = "[MASK]",
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[pd.DataFrame, str]:
    model, tokenizer, config, device, _ = load_demo_model()
    model.eval()

    candidates = _token_candidates(report, max_words=max_words)
    if not candidates:
        empty = pd.DataFrame(columns=["word", "importance", "probability_without_word", "start", "end"])
        return empty, highlighted_report(report, empty, max_words=max_words)

    if original_probability is None:
        _, image_tensor, input_ids, attention_mask = prepare_inputs(image_path, report, tokenizer, config, device)
        with torch.no_grad():
            original_probability = predict_probability(model, image_tensor, input_ids, attention_mask)

    rows = []
    total = len(candidates)
    for idx, candidate in enumerate(candidates, start=1):
        modified_report = _mask_word(report, candidate, mask_token=mask_token)
        _, image_tensor, input_ids, attention_mask = prepare_inputs(image_path, modified_report, tokenizer, config, device)
        with torch.no_grad():
            modified_probability = predict_probability(model, image_tensor, input_ids, attention_mask)
        rows.append(
            {
                "word": candidate.word,
                "importance": abs(float(original_probability) - float(modified_probability)),
                "probability_without_word": float(modified_probability),
                "start": candidate.start,
                "end": candidate.end,
            }
        )
        if progress_callback is not None:
            progress_callback(idx / total)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, highlighted_report(report, df, max_words=max_words)
    df = df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)
    return df, highlighted_report(report, df, max_words=max_words)

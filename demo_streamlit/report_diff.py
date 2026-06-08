from __future__ import annotations

import difflib
import html
import re


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", flags=re.MULTILINE)


NEGATIVE_EXPLANATIONS = {
    "distorted_negation": "Negation removes or flips a denial, creating a direct contradiction in the report.",
    "laterality_conflict": "Laterality changes left/right information while keeping the same image.",
    "temporal_mismatch": "Temporal wording is changed, for example improved versus worsened or stable versus progressed.",
    "partial_mismatch": "Severity or a local finding is changed while the rest of the report stays similar.",
    "pathology_semantic_swap": "A pathology term is replaced by a semantically different finding.",
    "random_report": "The report is replaced by another patient's report, producing an image-report mismatch.",
}


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text))


def _join_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in SENTENCE_RE.finditer(str(text)):
        start, end = match.span()
        if text[start:end].strip():
            spans.append((start, end))
    return spans or [(0, len(text))]


def _changed_sentence_window(original: str, modified: str, context_sentences: int = 1) -> tuple[str, str]:
    matcher = difflib.SequenceMatcher(a=original, b=modified)
    changed = [block for block in matcher.get_opcodes() if block[0] != "equal"]
    if not changed:
        return "", ""

    _tag, orig_start, orig_end, mod_start, mod_end = changed[0]

    def window(text: str, start: int, end: int) -> str:
        spans = _sentence_spans(text)
        indexes = [idx for idx, (s, e) in enumerate(spans) if not (e < start or s > end)]
        if not indexes:
            indexes = [0]
        left = max(0, min(indexes) - context_sentences)
        right = min(len(spans) - 1, max(indexes) + context_sentences)
        return text[spans[left][0]:spans[right][1]].strip()

    return window(original, orig_start, orig_end), window(modified, mod_start, mod_end)


def _highlight_inline(original: str, modified: str) -> str:
    original_tokens = _tokenize(original)
    modified_tokens = _tokenize(modified)
    matcher = difflib.SequenceMatcher(a=original_tokens, b=modified_tokens)
    parts: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(html.escape(_join_tokens(modified_tokens[j1:j2])))
            continue
        if tag in {"replace", "delete"} and i1 != i2:
            deleted = html.escape(_join_tokens(original_tokens[i1:i2]))
            if deleted:
                parts.append(
                    '<span style="background-color:#fee2e2;color:#7f1d1d;'
                    'padding:2px 4px;border-radius:4px;text-decoration:line-through;">'
                    f'{deleted}</span>'
                )
        if tag in {"replace", "insert"} and j1 != j2:
            added = html.escape(_join_tokens(modified_tokens[j1:j2]))
            if added:
                parts.append(
                    '<span style="background-color:#ffcc80;color:#111;'
                    'padding:2px 4px;border-radius:4px;font-weight:600;">'
                    f'{added}</span>'
                )

    return " ".join(part for part in parts if part).strip()


def _no_diff_html() -> str:
    return (
        '<div style="border:1px solid #d9dee7;border-radius:8px;padding:12px 14px;'
        'background:#fff7ed;color:#7c2d12;">'
        'Изменения не обнаружены или генератор вернул исходный текст.'
        '</div>'
    )


def build_report_diff_html(original_report: str, modified_report: str) -> str:
    original = str(original_report or "").strip()
    modified = str(modified_report or "").strip()
    if original == modified:
        return _no_diff_html()

    original_context, modified_context = _changed_sentence_window(original, modified, context_sentences=1)
    if not original_context and not modified_context:
        return _no_diff_html()

    highlighted = _highlight_inline(original_context, modified_context)
    return (
        '<div style="border:1px solid #d9dee7;border-radius:8px;padding:12px 14px;background:#ffffff;">'
        '<div style="font-weight:700;margin-bottom:8px;">Изменения в отчёте</div>'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;">'
        '<div style="border-left:3px solid #ef4444;padding-left:10px;">'
        '<div style="font-size:0.76rem;color:#64748b;font-weight:700;margin-bottom:4px;">Original report</div>'
        f'<div style="font-size:0.9rem;line-height:1.42;">{html.escape(original_context)}</div>'
        '</div>'
        '<div style="border-left:3px solid #f97316;padding-left:10px;">'
        '<div style="font-size:0.76rem;color:#64748b;font-weight:700;margin-bottom:4px;">Modified report</div>'
        f'<div style="font-size:0.9rem;line-height:1.42;">{html.escape(modified_context)}</div>'
        '</div>'
        '</div>'
        '<div style="font-size:0.76rem;color:#64748b;font-weight:700;margin-bottom:4px;">Highlighted diff</div>'
        f'<div style="font-size:0.94rem;line-height:1.55;background:#f8fafc;border-radius:6px;padding:10px;">{highlighted}</div>'
        '<div style="font-size:0.78rem;color:#64748b;margin-top:8px;">Red = removed/original text, orange = added or changed text.</div>'
        '</div>'
    )


def negative_pair_explanation(kind: str) -> str:
    return NEGATIVE_EXPLANATIONS.get(kind, "The generated report is intended to contradict the image-report pair.")

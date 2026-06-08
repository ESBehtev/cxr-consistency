from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from paths import PAIRS_CSV_PATH


REQUIRED_COLUMNS = [
    "image_path",
    "report",
    "label",
    "negative_type",
    "subject_id",
    "study_id",
]


@st.cache_data(show_spinner="Loading pair index...")
def load_pairs(csv_path: str = str(PAIRS_CSV_PATH)) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=lambda col: col in REQUIRED_COLUMNS)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Pairs CSV is missing required columns: {missing}")

    df = df[df["label"].astype(float).eq(1.0)].copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["study_id"] = df["study_id"].astype(str)
    df["report"] = df["report"].fillna("").astype(str)
    df["image_path"] = df["image_path"].astype(str)
    return df.reset_index(drop=True)


def row_to_example(row: pd.Series | dict) -> dict:
    data = dict(row)
    return {
        "image_path": str(data.get("image_path", "")),
        "report": str(data.get("report", "")),
        "subject_id": str(data.get("subject_id", "")),
        "study_id": str(data.get("study_id", "")),
        "negative_type": str(data.get("negative_type", "positive")),
    }


def get_example_by_index(df: pd.DataFrame, index: int) -> dict:
    if df.empty:
        raise ValueError("No positive examples found in the pairs CSV.")
    safe_index = int(index) % len(df)
    return row_to_example(df.iloc[safe_index]) | {"index": safe_index}


def random_index(df: pd.DataFrame, seed: int | None = None) -> int:
    if df.empty:
        raise ValueError("No examples available.")
    sample = df.sample(n=1, random_state=seed) if seed is not None else df.sample(n=1)
    return int(sample.index[0])


def find_by_study_id(df: pd.DataFrame, study_id: str) -> int | None:
    query = str(study_id).strip()
    if not query:
        return None
    matches = df.index[df["study_id"].eq(query)].tolist()
    return int(matches[0]) if matches else None


def find_by_patient_id(df: pd.DataFrame, patient_id: str) -> int | None:
    query = str(patient_id).strip()
    if not query:
        return None
    matches = df.index[df["subject_id"].eq(query)].tolist()
    return int(matches[0]) if matches else None


def resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path

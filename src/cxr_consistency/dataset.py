from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CXRConsistencyDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        tokenizer,
        split: str,
        image_size: int = 224,
        max_length: int = 256,
        max_samples: int | None = None,
        include_negative_types: list[str] | None = None,
        max_positives_per_negative: float | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer
        self.split = split
        self.image_size = image_size
        self.max_length = max_length

        df = pd.read_csv(self.csv_path)
        df = df[df["split"] == split].reset_index(drop=True)

        if include_negative_types:
            include_negative_types = set(include_negative_types)
            is_positive = df["label"].astype(float) == 1.0
            is_included_negative = (
                (df["label"].astype(float) == 0.0)
                & df["negative_type"].isin(include_negative_types)
            )
            df = df[is_positive | is_included_negative].reset_index(drop=True)

        if max_positives_per_negative is not None:
            positives = df[df["label"].astype(float) == 1.0]
            negatives = df[df["label"].astype(float) == 0.0]
            max_positives = int(len(negatives) * float(max_positives_per_negative))

            if max_positives > 0 and len(positives) > max_positives:
                positives = positives.sample(
                    n=max_positives,
                    random_state=42,
                )
                df = pd.concat([positives, negatives], ignore_index=True)
                df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        if max_samples is not None:
            df = df.sample(
                n=min(max_samples, len(df)),
                random_state=42,
            ).reset_index(drop=True)

        self.df = df

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        report = str(row["report"])
        label = float(row["label"])

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        encoded = self.tokenizer(
            report,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32),
            "negative_type": row["negative_type"],
        }

        return item

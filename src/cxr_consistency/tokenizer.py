import hashlib
import re

import torch


class SimpleHashTokenizer:
    def __init__(
        self,
        vocab_size: int = 30522,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id

    def __len__(self) -> int:
        return self.vocab_size

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+|[^\s\w]", str(text).lower())

    def token_to_id(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % (self.vocab_size - 2) + 2

    def __call__(
        self,
        text: str,
        padding: str | bool = False,
        truncation: bool = True,
        max_length: int = 256,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor | list[int]]:
        token_ids = [self.token_to_id(token) for token in self.tokenize(text)]

        if truncation:
            token_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)

        if padding == "max_length" and len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids = token_ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([token_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
        }

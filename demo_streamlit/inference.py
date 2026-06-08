from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import torch
import yaml
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from paths import CHECKPOINT_PATH, CONFIG_PATH, MODEL_SOURCE_PATH, require_file
from paths import PROJECT_ROOT  # noqa: F401 - importing sets stable project-relative context
from cxr_consistency.model import CXRConsistencyModel, get_device
from cxr_consistency.tokenizer import SimpleHashTokenizer


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    require_file(config_path, "Config")
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def build_tokenizer(config: dict):
    tokenizer_name = config["tokenizer_name"]
    if tokenizer_name == "simple":
        return SimpleHashTokenizer(vocab_size=int(config.get("tokenizer_vocab_size", 30522)))
    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        local_files_only=bool(config.get("local_files_only", False)),
    )


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format; expected a state_dict-like object.")
    return {str(k).removeprefix("module."): v for k, v in checkpoint.items()}


@st.cache_resource(show_spinner="Loading ConvNeXt Tiny + CXR-BERT...")
def load_demo_model(checkpoint_path: str = str(CHECKPOINT_PATH), config_path: str = str(CONFIG_PATH)):
    checkpoint_path_obj = Path(checkpoint_path)
    config_path_obj = Path(config_path)
    require_file(checkpoint_path_obj, "Checkpoint")
    config = load_config(config_path_obj)
    tokenizer = build_tokenizer(config)

    model = CXRConsistencyModel(
        image_encoder_name=config.get("image_encoder_name", "convnext_tiny"),
        text_encoder_name=config.get("text_encoder_name", "cxrbert"),
        pretrained_image=bool(config.get("pretrained_image", False)),
        freeze_image_encoder=bool(config.get("freeze_image_encoder", False)),
        freeze_text_encoder=bool(config.get("freeze_text_encoder", False)),
        tokenizer_vocab_size=len(tokenizer) if hasattr(tokenizer, "__len__") else int(config.get("tokenizer_vocab_size", 30522)),
        pad_token_id=getattr(tokenizer, "pad_token_id", 0) or 0,
        projection_dim=int(config.get("projection_dim", 256)),
        dropout=float(config.get("dropout", 0.2)),
    )

    checkpoint = torch.load(checkpoint_path_obj, map_location="cpu")
    model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=True)
    device = get_device()
    model.to(device)
    model.eval()
    return model, tokenizer, config, device, str(MODEL_SOURCE_PATH)


def image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
    ])


def prepare_inputs(image_path: Path, report: str, tokenizer, config: dict, device: torch.device):
    image_size = int(config.get("image_size", 224))
    max_length = int(config.get("max_length", 256))
    pil_image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image_size)(pil_image).unsqueeze(0).to(device)
    encoded = tokenizer(
        str(report),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    return pil_image, image_tensor, input_ids, attention_mask


@torch.no_grad()
def predict_probability(model, image_tensor: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
    model.eval()
    logits = model(image=image_tensor, input_ids=input_ids, attention_mask=attention_mask)
    return float(torch.sigmoid(logits).detach().cpu().item())


def run_inference(image_path: Path, report: str) -> dict[str, Any]:
    model, tokenizer, config, device, model_source = load_demo_model()
    pil_image, image_tensor, input_ids, attention_mask = prepare_inputs(image_path, report, tokenizer, config, device)
    probability = predict_probability(model, image_tensor, input_ids, attention_mask)
    return {
        "probability": probability,
        "pil_image": pil_image,
        "model_source": model_source,
        "device": str(device),
    }

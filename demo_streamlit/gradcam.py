from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from inference import load_demo_model, prepare_inputs


def _last_conv_module(model: torch.nn.Module) -> torch.nn.Module | None:
    last = None
    for module in model.image_encoder.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module
    return last


def _overlay_heatmap(image: Image.Image, cam: np.ndarray) -> Image.Image:
    base = image.convert("RGB").resize((cam.shape[1], cam.shape[0]))
    base_arr = np.asarray(base).astype(np.float32) / 255.0
    heat = np.zeros_like(base_arr)
    heat[..., 0] = cam
    heat[..., 1] = np.clip(1.0 - np.abs(cam - 0.5) * 2.0, 0.0, 1.0) * 0.35
    blended = np.clip(base_arr * 0.58 + heat * 0.42, 0.0, 1.0)
    return Image.fromarray((blended * 255).astype(np.uint8))


def build_gradcam(image_path: Path, report: str) -> tuple[Image.Image | None, Image.Image | None, str | None]:
    try:
        model, tokenizer, config, device, _ = load_demo_model()
        pil_image, image_tensor, input_ids, attention_mask = prepare_inputs(image_path, report, tokenizer, config, device)
        target_layer = _last_conv_module(model)
        if target_layer is None:
            return pil_image, None, "No convolutional layer was found for Grad-CAM."

        activations: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []

        def forward_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            activations.append(output.detach())

        def backward_hook(_module: torch.nn.Module, _grad_input: tuple[Any, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
            gradients.append(grad_output[0].detach())

        handle_fwd = target_layer.register_forward_hook(forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(backward_hook)
        try:
            model.zero_grad(set_to_none=True)
            logits = model(image=image_tensor, input_ids=input_ids, attention_mask=attention_mask)
            logits.sum().backward()
        finally:
            handle_fwd.remove()
            handle_bwd.remove()

        if not activations or not gradients:
            return pil_image, None, "Grad-CAM hooks did not capture activations."

        acts = activations[-1]
        grads = gradients[-1]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=pil_image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return pil_image, _overlay_heatmap(pil_image, cam_np), None
    except Exception as exc:  # UI must survive Grad-CAM failures during demo.
        try:
            return Image.open(image_path).convert("RGB"), None, str(exc)
        except Exception:
            return None, None, str(exc)

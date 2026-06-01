from pathlib import Path

import mlflow
import numpy as np
import torch
from cxr_consistency.metrics import compute_binary_metrics, compute_metrics_with_best_threshold
from tqdm import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def add_prediction_stats(
    metrics: dict,
    labels,
    logits,
    probs,
) -> None:
    labels = np.asarray(labels).astype(int)
    logits = np.asarray(logits).astype(float)
    probs = np.asarray(probs).astype(float)

    for name, values in [("logits", logits), ("probs", probs)]:
        metrics[f"{name}_mean"] = float(np.mean(values))
        metrics[f"{name}_std"] = float(np.std(values))
        metrics[f"{name}_min"] = float(np.min(values))
        metrics[f"{name}_max"] = float(np.max(values))

        for label in [0, 1]:
            mask = labels == label
            prefix = f"{name}_label{label}"

            if not np.any(mask):
                metrics[f"{prefix}_mean"] = 0.0
                metrics[f"{prefix}_std"] = 0.0
                metrics[f"{prefix}_min"] = 0.0
                metrics[f"{prefix}_max"] = 0.0
                continue

            group = values[mask]
            metrics[f"{prefix}_mean"] = float(np.mean(group))
            metrics[f"{prefix}_std"] = float(np.std(group))
            metrics[f"{prefix}_min"] = float(np.min(group))
            metrics[f"{prefix}_max"] = float(np.max(group))

    preds = (probs >= 0.5).astype(int)
    metrics["pred_positive_fraction"] = float(np.mean(preds))


def add_grouped_negative_type_metrics(
    metrics: dict,
    labels,
    probs,
    negative_types,
) -> None:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    negative_types = np.asarray(negative_types).astype(str)

    for negative_type in sorted(set(negative_types.tolist()) - {"positive"}):
        mask = (labels == 1) | ((labels == 0) & (negative_types == negative_type))
        group_labels = labels[mask]
        group_probs = probs[mask]

        count_positive = int(np.sum(group_labels == 1))
        count_negative = int(np.sum(group_labels == 0))

        if count_positive == 0 or count_negative == 0:
            continue

        group_metrics = compute_binary_metrics(group_labels, group_probs)
        safe_type = negative_type.replace("/", "_").replace(" ", "_")
        prefix = f"by_type_{safe_type}"

        metrics[f"{prefix}_accuracy"] = group_metrics["accuracy"]
        metrics[f"{prefix}_f1"] = group_metrics["f1"]
        metrics[f"{prefix}_roc_auc"] = group_metrics["roc_auc"]
        metrics[f"{prefix}_pr_auc"] = group_metrics["pr_auc"]
        metrics[f"{prefix}_count_positive"] = float(count_positive)
        metrics[f"{prefix}_count_negative"] = float(count_negative)


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler=None,
    scheduler=None,
    epoch: int = 0,
    log_every_steps: int = 10,
    log_batch_loss: bool = False,
    grad_clip_norm: float | None = None,
):
    model.train()

    total_loss = 0.0
    grad_norms = []
    all_labels = []
    all_logits = []
    all_probs = []
    all_negative_types = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress_bar = tqdm(dataloader)

    for step, batch in enumerate(progress_bar):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        use_amp = scaler is not None and device.type == "cuda"

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(
                    image=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(grad_clip_norm) if grad_clip_norm is not None else float("inf"),
            )
            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(
                image=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = criterion(logits, labels)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(grad_clip_norm) if grad_clip_norm is not None else float("inf"),
            )
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        probs = torch.sigmoid(logits)

        total_loss += float(loss.item())
        grad_norms.append(float(grad_norm.detach().cpu()))

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_logits.extend(logits.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_negative_types.extend(batch["negative_type"])

        global_step = epoch * len(dataloader) + step + 1

        if log_batch_loss and step % log_every_steps == 0:
            mlflow.log_metric(
                key="train_batch_loss",
                value=float(loss.item()),
                step=global_step,
                synchronous=True,
            )

        progress_bar.set_description(
            f"train_loss={loss.item():.4f}"
        )

    metrics = compute_metrics_with_best_threshold(all_labels, all_probs)
    add_prediction_stats(metrics, all_labels, all_logits, all_probs)
    add_grouped_negative_type_metrics(metrics, all_labels, all_probs, all_negative_types)
    metrics["loss"] = float(total_loss / len(dataloader))
    metrics["grad_norm_mean"] = float(np.mean(grad_norms))
    metrics["grad_norm_max"] = float(np.max(grad_norms))
    metrics["lr"] = float(optimizer.param_groups[0]["lr"])

    if device.type == "cuda":
        metrics["cuda_peak_memory_mb"] = float(
            torch.cuda.max_memory_allocated(device) / 1024 / 1024
        )

    return metrics


@torch.no_grad()
def validate_epoch(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_logits = []
    all_probs = []
    all_negative_types = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress_bar = tqdm(dataloader)

    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)

        images = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        logits = model(
            image=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        total_loss += float(loss.item())

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_logits.extend(logits.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_negative_types.extend(batch["negative_type"])

        progress_bar.set_description(
            f"valid_loss={loss.item():.4f}"
        )

    metrics = compute_metrics_with_best_threshold(all_labels, all_probs)
    add_prediction_stats(metrics, all_labels, all_logits, all_probs)
    add_grouped_negative_type_metrics(metrics, all_labels, all_probs, all_negative_types)
    metrics["loss"] = float(total_loss / len(dataloader))

    if device.type == "cuda":
        metrics["cuda_peak_memory_mb"] = float(
            torch.cuda.max_memory_allocated(device) / 1024 / 1024
        )

    return metrics


def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    output_dir,
    filename="best_model.pt",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    path = output_dir / filename
    torch.save(checkpoint, path)

    print(f"Saved checkpoint to {path}")


def print_metrics(prefix: str, metrics: dict):
    print(
        f"{prefix} | "
        f"loss={metrics['loss']:.4f} | "
        f"acc={metrics['accuracy']:.4f} | "
        f"f1@0.5={metrics['f1']:.4f} | "
        f"best_f1={metrics['best_f1']:.4f} | "
        f"best_thr={metrics['best_threshold']:.2f} | "
        f"auc={metrics['roc_auc']:.4f} | "
        f"pr_auc={metrics['pr_auc']:.4f}"
    )
    extra = []
    if "pred_positive_fraction" in metrics:
        extra.append(f"pred_pos={metrics['pred_positive_fraction']:.4f}")
    if "grad_norm_mean" in metrics:
        extra.append(
            f"grad_norm={metrics['grad_norm_mean']:.4f}/{metrics['grad_norm_max']:.4f}"
        )
    if "lr" in metrics:
        extra.append(f"lr={metrics['lr']:.2e}")
    if "cuda_peak_memory_mb" in metrics:
        extra.append(f"cuda_peak={metrics['cuda_peak_memory_mb']:.0f}MB")
    if extra:
        print(f"{prefix} diag | " + " | ".join(extra))

    print(
        f"{prefix} logits | "
        f"mean={metrics['logits_mean']:.4f} "
        f"std={metrics['logits_std']:.4f} "
        f"min={metrics['logits_min']:.4f} "
        f"max={metrics['logits_max']:.4f} | "
        f"y0=({metrics['logits_label0_mean']:.4f},"
        f"{metrics['logits_label0_std']:.4f},"
        f"{metrics['logits_label0_min']:.4f},"
        f"{metrics['logits_label0_max']:.4f}) "
        f"y1=({metrics['logits_label1_mean']:.4f},"
        f"{metrics['logits_label1_std']:.4f},"
        f"{metrics['logits_label1_min']:.4f},"
        f"{metrics['logits_label1_max']:.4f})"
    )
    print(
        f"{prefix} probs  | "
        f"mean={metrics['probs_mean']:.4f} "
        f"std={metrics['probs_std']:.4f} "
        f"min={metrics['probs_min']:.4f} "
        f"max={metrics['probs_max']:.4f} | "
        f"y0=({metrics['probs_label0_mean']:.4f},"
        f"{metrics['probs_label0_std']:.4f},"
        f"{metrics['probs_label0_min']:.4f},"
        f"{metrics['probs_label0_max']:.4f}) "
        f"y1=({metrics['probs_label1_mean']:.4f},"
        f"{metrics['probs_label1_std']:.4f},"
        f"{metrics['probs_label1_min']:.4f},"
        f"{metrics['probs_label1_max']:.4f})"
    )

    grouped_prefixes = sorted(
        {
            key.rsplit("_", 1)[0]
            for key in metrics
            if key.startswith("by_type_") and key.endswith("_accuracy")
        }
    )

    if grouped_prefixes:
        print(f"{prefix} grouped by negative_type:")

    for group_prefix in grouped_prefixes:
        negative_type = group_prefix.replace("by_type_", "", 1)
        print(
            f"  {negative_type}: "
            f"acc={metrics[f'{group_prefix}_accuracy']:.4f} "
            f"f1={metrics[f'{group_prefix}_f1']:.4f} "
            f"auc={metrics[f'{group_prefix}_roc_auc']:.4f} "
            f"pr_auc={metrics[f'{group_prefix}_pr_auc']:.4f} "
            f"pos={int(metrics[f'{group_prefix}_count_positive'])} "
            f"neg={int(metrics[f'{group_prefix}_count_negative'])}"
        )


def log_metrics_to_mlflow(metrics: dict, step: int, prefix: str):
    for metric_name, value in metrics.items():
        mlflow.log_metric(
            key=f"{prefix}_{metric_name}",
            value=float(value),
            step=int(step),
            synchronous=True,
        )

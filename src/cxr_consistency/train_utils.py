from pathlib import Path

import mlflow
import torch
from cxr_consistency.metrics import compute_metrics_with_best_threshold
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


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler=None,
    epoch: int = 0,
    log_every_steps: int = 10,
    log_batch_loss: bool = False,
):
    model.train()

    total_loss = 0.0
    all_labels = []
    all_probs = []

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
            optimizer.step()

        probs = torch.sigmoid(logits)

        total_loss += float(loss.item())

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

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
    metrics["loss"] = float(total_loss / len(dataloader))

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
    all_probs = []

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
        all_probs.extend(probs.detach().cpu().numpy().tolist())

        progress_bar.set_description(
            f"valid_loss={loss.item():.4f}"
        )

    metrics = compute_metrics_with_best_threshold(all_labels, all_probs)
    metrics["loss"] = float(total_loss / len(dataloader))

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


def log_metrics_to_mlflow(metrics: dict, step: int, prefix: str):
    for metric_name, value in metrics.items():
        mlflow.log_metric(
            key=f"{prefix}_{metric_name}",
            value=float(value),
            step=int(step),
            synchronous=True,
        )
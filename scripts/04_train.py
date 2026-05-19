from pathlib import Path
import argparse
import yaml

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cxr_consistency.dataset import CXRConsistencyDataset
from cxr_consistency.model import CXRConsistencyModel, get_device
from cxr_consistency.train_utils import (
    train_epoch,
    validate_epoch,
    save_checkpoint,
    print_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_pos_weight(csv_path: str) -> torch.Tensor:
    import pandas as pd

    df = pd.read_csv(csv_path)

    positives = (df["label"] == 1).sum()
    negatives = (df["label"] == 0).sum()
    pos_weight = negatives / max(positives, 1)

    print(f"\nPositives: {positives}")
    print(f"Negatives: {negatives}")
    print(f"pos_weight: {pos_weight:.4f}")

    return torch.tensor(pos_weight, dtype=torch.float32)


def init_history() -> dict:
    return {"epoch": []}


def update_history(
    history: dict,
    train_metrics: dict,
    valid_metrics: dict,
    epoch: int,
) -> None:
    history["epoch"].append(epoch + 1)

    for key, value in train_metrics.items():
        history.setdefault(f"train_{key}", []).append(float(value))

    for key, value in valid_metrics.items():
        history.setdefault(f"valid_{key}", []).append(float(value))


def log_epoch_metrics(
    train_metrics: dict,
    valid_metrics: dict,
    epoch: int,
    model_id: str | None = None,
) -> None:
    step = epoch + 1

    metrics_to_log = {}

    for key, value in train_metrics.items():
        metrics_to_log[f"train_{key}"] = float(value)

    for key, value in valid_metrics.items():
        metrics_to_log[f"valid_{key}"] = float(value)

    metrics_to_log["epoch"] = float(step)

    mlflow.log_metrics(
        metrics=metrics_to_log,
        step=step,
        model_id=model_id,
        synchronous=True,
    )

    run = mlflow.active_run()
    print(f"Logged epoch metrics to MLflow run_id={run.info.run_id}")

    if model_id is not None:
        print(f"Logged model metrics to MLflow model_id={model_id}")

    print("Logged metrics:", ", ".join(metrics_to_log.keys()))


def plot_training_curves(history: dict, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("loss", "Loss"),
        ("f1", "F1@0.5"),
        ("best_f1", "Best F1"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("best_threshold", "Best Threshold"),
    ]

    epochs = history["epoch"]

    for metric_key, title in plot_specs:
        train_key = f"train_{metric_key}"
        valid_key = f"valid_{metric_key}"

        if train_key not in history or valid_key not in history:
            continue

        plt.figure()
        plt.plot(epochs, history[train_key], marker="o", label=train_key)
        plt.plot(epochs, history[valid_key], marker="o", label=valid_key)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.grid(True)

        path = plots_dir / f"{metric_key}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(str(path), artifact_path="plots")


def log_history_table(history: dict) -> None:
    import pandas as pd

    df = pd.DataFrame(history)

    mlflow.log_table(
        data=df,
        artifact_file="metrics_history.json",
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    mlflow.set_tracking_uri("sqlite:///mlflow_clean.db")
    mlflow.set_experiment(config["experiment_name"])

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

    train_dataset = CXRConsistencyDataset(
        csv_path=config["pairs_csv"],
        tokenizer=tokenizer,
        split="train",
        image_size=config["image_size"],
        max_length=config["max_length"],
        max_samples=config.get("max_train_samples"),
    )

    valid_dataset = CXRConsistencyDataset(
        csv_path=config["pairs_csv"],
        tokenizer=tokenizer,
        split="valid",
        image_size=config["image_size"],
        max_length=config["max_length"],
        max_samples=config.get("max_valid_samples"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=False,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=False,
    )

    model = CXRConsistencyModel(
        image_encoder_name=config["image_encoder_name"],
        text_encoder_name=config["text_encoder_name"],
        pretrained_image=config["pretrained_image"],
        freeze_image_encoder=config["freeze_image_encoder"],
        freeze_text_encoder=config["freeze_text_encoder"],
        projection_dim=config["projection_dim"],
        dropout=config["dropout"],
    ).to(device)

    pos_weight = build_pos_weight(config["pairs_csv"]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    experiment_dir = Path(config["experiment_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    history = init_history()

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        logged_model = mlflow.pytorch.log_model(
            model,
            name="model",
        )
        model_id = logged_model.model_id

        mlflow.log_params(config)

        mlflow.set_tags(
            {
                "project": "cxr_consistency",
                "task": "multimodal_consistency",
                "image_encoder": config["image_encoder_name"],
                "text_encoder": config["text_encoder_name"],
            }
        )

        for epoch in range(config["epochs"]):
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")

            train_metrics = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                epoch=epoch,
                log_every_steps=config.get("log_every_steps", 50),
                log_batch_loss=config.get("log_batch_loss", False),
            )

            valid_metrics = validate_epoch(
                model=model,
                dataloader=valid_loader,
                criterion=criterion,
                device=device,
            )

            print_metrics("TRAIN", train_metrics)
            print_metrics("VALID", valid_metrics)

            step = epoch + 1

            metrics_to_log = {}

            for key, value in train_metrics.items():
                metrics_to_log[f"train_{key}"] = float(value)

            for key, value in valid_metrics.items():
                metrics_to_log[f"valid_{key}"] = float(value)

            metrics_to_log["epoch"] = float(step)

            for key, value in metrics_to_log.items():
                mlflow.log_metric(
                    key=key,
                    value=value,
                    step=step,
                    model_id=model_id,
                    synchronous=True,
                )

            print(f"Logged epoch metrics to MLflow run_id={run_id}")
            print("Logged metrics:", ", ".join(metrics_to_log.keys()))

            update_history(
                history=history,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                epoch=epoch,
            )

            log_history_table(history)

            plot_training_curves(
                history=history,
                output_dir=experiment_dir,
            )

            current_f1 = valid_metrics.get("best_f1", valid_metrics["f1"])

            if current_f1 > best_f1:
                best_f1 = current_f1

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=valid_metrics,
                    output_dir=experiment_dir,
                    filename="best_model.pt",
                )

                mlflow.log_metric(
                    key="best_valid_f1",
                    value=float(best_f1),
                    step=step,
                    model_id=model_id,
                    synchronous=True,
                )

        print(f"\nBest validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
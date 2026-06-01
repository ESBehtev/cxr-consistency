from pathlib import Path
import argparse
import os
import sys
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))
MPLCONFIGDIR = ROOT_DIR / "logs" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
XDG_CACHE_HOME = ROOT_DIR / "logs" / "cache"
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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
from cxr_consistency.tokenizer import SimpleHashTokenizer
from cxr_consistency.train_utils import (
    train_epoch,
    validate_epoch,
    save_checkpoint,
    print_metrics,
)


def resolve_device(config: dict) -> torch.device:
    requested_device = config.get("device")

    if requested_device is None or requested_device == "auto":
        return get_device()

    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Config requested device=mps, but PyTorch MPS is not available")
        return torch.device("mps")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested device=cuda, but CUDA is not available")
        return torch.device("cuda")

    if requested_device == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {requested_device}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_mlflow_tracking(config: dict) -> None:
    tracking_uri = config.get("mlflow_tracking_uri", "sqlite:///mlflow_clean.db")

    if tracking_uri.startswith("sqlite:///"):
        db_path = Path(tracking_uri.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["experiment_name"])


def flatten_config(config: dict, prefix: str = "") -> dict:
    flat = {}

    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flat.update(flatten_config(value, full_key))
        else:
            flat[full_key] = value

    return flat


def build_tokenizer(config: dict):
    tokenizer_name = config["tokenizer_name"]

    if tokenizer_name == "simple":
        return SimpleHashTokenizer(
            vocab_size=int(config.get("tokenizer_vocab_size", 30522)),
        )

    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.get("trust_remote_code", False),
        local_files_only=config.get("local_files_only", False),
    )




def build_scheduler(config: dict, optimizer, steps_per_epoch: int):
    scheduler_type = config.get("scheduler_type")

    if not scheduler_type or scheduler_type == "none":
        return None

    total_steps = int(steps_per_epoch * int(config["epochs"]))
    warmup_steps = int(config.get("warmup_steps", 0))

    if warmup_steps <= 0:
        warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.0)))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)

        if scheduler_type == "cosine":
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()

        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    print(
        f"Using scheduler={scheduler_type}, total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}"
    )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def use_pos_weight(config: dict) -> bool:
    loss_config = config.get("loss", {})

    if isinstance(loss_config, dict):
        return bool(loss_config.get("use_pos_weight", True))

    return bool(config.get("use_pos_weight", True))


def build_pos_weight(labels) -> torch.Tensor:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    pos_weight = negatives / max(positives, 1)

    print(f"\nPositives: {positives}")
    print(f"Negatives: {negatives}")
    print(f"pos_weight: {pos_weight:.4f}")

    return torch.tensor(pos_weight, dtype=torch.float32)


def validate_labels(dataset: CXRConsistencyDataset, name: str) -> None:
    labels = dataset.df["label"]
    unique_labels = sorted(labels.dropna().unique().tolist())
    invalid = sorted(set(unique_labels) - {0, 1, 0.0, 1.0})

    print(f"\n{name} labels:")
    print(labels.value_counts(dropna=False).sort_index().to_string())
    print(f"{name} positive fraction: {(labels == 1).mean():.4f}")

    if labels.isna().any():
        raise ValueError(f"{name} labels contain NaN values")

    if invalid:
        raise ValueError(f"{name} labels must be 0/1, got: {invalid}")

    if not labels.isin([0, 1, 0.0, 1.0]).all():
        raise ValueError(f"{name} labels contain values outside 0/1")

    if "negative_type" in dataset.df.columns:
        positive_type = dataset.df["negative_type"].astype(str) == "positive"
        positive_label = labels.astype(float) == 1.0

        if not positive_type.equals(positive_label):
            bad = dataset.df.loc[
                positive_type != positive_label,
                ["image_path", "report", "label", "negative_type"],
            ].head(5)
            raise ValueError(
                f"{name} labels look inverted or inconsistent with negative_type:\n"
                f"{bad.to_string(index=False)}"
            )


def print_dataset_summary(dataset: CXRConsistencyDataset, name: str) -> None:
    labels = dataset.df["label"].astype(float)

    print(f"\n{name} dataset after filters:")
    print(f"  total: {len(dataset.df)}")
    print(f"  positives: {int((labels == 1.0).sum())}")
    print(f"  negatives: {int((labels == 0.0).sum())}")

    if "negative_type" in dataset.df.columns:
        print("  negative_type counts:")
        print(
            dataset.df["negative_type"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

    if "split" in dataset.df.columns:
        print("  split counts:")
        print(dataset.df["split"].value_counts(dropna=False).sort_index().to_string())


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def print_trainable_parameters(model: CXRConsistencyModel) -> None:
    blocks = {
        "image_encoder": model.image_encoder,
        "text_encoder": model.text_encoder,
        "classifier/fusion": torch.nn.ModuleList(
            [
                model.image_projection,
                model.text_projection,
                model.classifier,
            ]
        ),
    }

    print("\nTrainable parameters:")
    total_all = 0
    trainable_all = 0

    for name, module in blocks.items():
        total, trainable = count_parameters(module)
        total_all += total
        trainable_all += trainable
        print(f"  {name}: trainable={trainable:,} / total={total:,}")

    print(f"  total: trainable={trainable_all:,} / total={total_all:,}")

    if trainable_all == 0:
        raise ValueError("Optimizer would receive zero trainable parameters")


def check_tokenizer_diversity(
    tokenizer,
    dataset: CXRConsistencyDataset,
    max_length: int,
    sample_size: int = 16,
) -> None:
    sample = dataset.df.head(min(sample_size, len(dataset.df)))
    sequences = []
    unique_token_ids = set()
    nonpad_lengths = []

    for report in sample["report"]:
        encoded = tokenizer(
            str(report),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        nonpad = [
            token_id
            for token_id, mask in zip(input_ids, attention_mask)
            if int(mask) == 1
        ]
        sequences.append(tuple(nonpad))
        unique_token_ids.update(nonpad)
        nonpad_lengths.append(len(nonpad))

    unique_sequences = len(set(sequences))

    print("\nTokenizer sanity:")
    print(f"  sampled reports: {len(sample)}")
    print(f"  unique non-pad sequences: {unique_sequences}")
    print(f"  unique token ids: {len(unique_token_ids)}")
    print(f"  non-pad lengths: {nonpad_lengths[:8]}")
    if sequences:
        print(f"  first sequence head: {list(sequences[0][:16])}")

    if len(sample) > 1 and unique_sequences <= 1:
        raise ValueError("Tokenizer produced identical token id sequences for sampled reports")


def log_metric(
    key: str,
    value: float,
    step: int,
    model_id: str | None,
) -> None:
    kwargs = {
        "key": key,
        "value": float(value),
        "step": int(step),
        "synchronous": True,
    }

    if model_id is not None:
        kwargs["model_id"] = model_id

    mlflow.log_metric(**kwargs)


def get_data_config(config: dict) -> dict:
    data_config = config.get("data", {})
    return data_config if isinstance(data_config, dict) else {}


def get_include_negative_types(config: dict) -> list[str] | None:
    negative_types = get_data_config(config).get("include_negative_types")

    if not negative_types:
        return None

    return [str(negative_type) for negative_type in negative_types]


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

    prepare_mlflow_tracking(config)

    device = resolve_device(config)
    print(f"Using device: {device}")

    tokenizer = build_tokenizer(config)
    data_config = get_data_config(config)
    include_negative_types = get_include_negative_types(config)
    max_positives_per_negative = data_config.get("max_positives_per_negative")

    train_dataset = CXRConsistencyDataset(
        csv_path=config["pairs_csv"],
        tokenizer=tokenizer,
        split="train",
        image_size=config["image_size"],
        max_length=config["max_length"],
        max_samples=config.get("max_train_samples"),
        include_negative_types=include_negative_types,
        max_positives_per_negative=max_positives_per_negative,
    )

    valid_dataset = CXRConsistencyDataset(
        csv_path=config["pairs_csv"],
        tokenizer=tokenizer,
        split=config.get("valid_split", "valid"),
        image_size=config["image_size"],
        max_length=config["max_length"],
        max_samples=config.get("max_valid_samples"),
        include_negative_types=include_negative_types,
        max_positives_per_negative=max_positives_per_negative,
    )

    print_dataset_summary(train_dataset, "train")
    print_dataset_summary(valid_dataset, "valid")
    validate_labels(train_dataset, "train")
    validate_labels(valid_dataset, "valid")
    check_tokenizer_diversity(
        tokenizer=tokenizer,
        dataset=train_dataset,
        max_length=config["max_length"],
        sample_size=config.get("tokenizer_debug_samples", 16),
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
        tokenizer_vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id or 0,
        projection_dim=config["projection_dim"],
        dropout=config["dropout"],
    ).to(device)

    print_trainable_parameters(model)

    if use_pos_weight(config):
        pos_weight = build_pos_weight(train_dataset.df["label"]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        print("\nUsing BCEWithLogitsLoss without pos_weight")
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    scheduler = build_scheduler(config, optimizer, len(train_loader))

    use_amp = bool(config.get("amp", True))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" and use_amp else None
    print(f"AMP enabled: {scaler is not None}")

    experiment_dir = Path(config["experiment_dir"])
    experiment_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    history = init_history()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        model_id = None

        if config.get("log_model", False):
            logged_model = mlflow.pytorch.log_model(
                model,
                name="model",
            )
            model_id = logged_model.model_id

        mlflow.log_params(flatten_config(config))

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
                scheduler=scheduler,
                epoch=epoch,
                log_every_steps=config.get("log_every_steps", 50),
                log_batch_loss=config.get("log_batch_loss", False),
                grad_clip_norm=config.get("grad_clip_norm"),
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
                log_metric(
                    key=key,
                    value=value,
                    step=step,
                    model_id=model_id,
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

                if config.get("save_checkpoints", True):
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics=valid_metrics,
                        output_dir=experiment_dir,
                        filename="best_model.pt",
                    )

                log_metric(
                    key="best_valid_f1",
                    value=float(best_f1),
                    step=step,
                    model_id=model_id,
                )

        print(f"\nBest validation F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()

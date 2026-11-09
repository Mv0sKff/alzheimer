from dataclasses import asdict

import torch
from torch import nn, optim

from schemas import History, selection_score
from utils import (
    compute_metrics,
    get_rng_state,
    plot_metric,
    set_rng_state,
    write_history,
)


def train_model(model, train_loader, val_loader, fold, output_dir, args, save=False):
    train_history = History()
    val_history = History()
    lrs = []
    best_score = (-1.0, -1.0)
    epochs_without_improvement = 0
    start_epoch = 0
    device = next(model.parameters()).device
    use_amp = args.amp and device.type == "cuda"
    num_classes = model.model.fc.out_features
    fold_labels = torch.tensor(train_loader.dataset.dataset["Label"].to_numpy())
    counts = torch.bincount(fold_labels, minlength=num_classes).float()
    class_weights = counts.sum() / (num_classes * counts.clamp(min=1))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    last_path = output_dir / f"last_fold_{fold}.pt"
    history_path = output_dir / f"history_fold_{fold}.csv"
    if last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        train_history = History(**state["train_history"])
        val_history = History(**state["val_history"])
        lrs = state["lrs"]
        best_score = tuple(state["best_score"])
        epochs_without_improvement = state["epochs_without_improvement"]
        set_rng_state(state["rng"], train_loader.generator)
        start_epoch = state["epoch"] + 1
        print(f"fold {fold}: resuming at epoch {start_epoch}")
    for epoch in range(start_epoch, args.epochs):
        if epochs_without_improvement >= args.patience:
            print(
                f"fold {fold}: early stopping, no improvement for {args.patience} epochs"
            )
            break

        model.train()
        running_loss = 0.0
        total = 0
        train_logits = []
        train_labels = []
        train_subjects = []

        for batch in train_loader:
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            # calc more metrics if needed

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            train_logits.append(outputs.detach().float().cpu())
            train_labels.append(labels.cpu())
            train_subjects.extend(batch["subject"])

        train_history.append(
            running_loss / total,
            *compute_metrics(
                torch.cat(train_labels), torch.cat(train_logits), train_subjects
            ),
        )
        val_history.append(*validate_model(model, val_loader, criterion, use_amp))

        val_loss = val_history.loss[-1]
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
        if selection_score(val_history) > best_score:
            best_score = selection_score(val_history)
            epochs_without_improvement = 0
            if save:
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(), output_dir / f"best_model_fold_{fold}.pt"
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement % args.lr_patience == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.1

        if save:
            output_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = last_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "train_history": asdict(train_history),
                    "val_history": asdict(val_history),
                    "lrs": lrs,
                    "best_score": list(best_score),
                    "epochs_without_improvement": epochs_without_improvement,
                    "rng": get_rng_state(train_loader.generator),
                    "epoch": epoch,
                },
                tmp_path,
            )
            tmp_path.replace(last_path)
            write_history(history_path, lrs, train_history, val_history)

        print(
            f"epoch: {epoch} |",
            f"lr: {lr:.1e} |",
            f"train loss: {train_history.loss[-1]:.4f} |",
            f"val loss: {val_loss:.4f} |",
            f"val balanced accuracy: {val_history.balanced_accuracy[-1]:.4f} |",
            f"val ROC-AUC: {val_history.roc_auc[-1]:.4f} |",
            f"val subject accuracy: {val_history.subject_accuracy[-1]:.4f} |",
            "val subject balanced accuracy:",
            f"{val_history.subject_balanced_accuracy[-1]:.4f} |",
            f"val subject ROC-AUC: {val_history.subject_roc_auc[-1]:.4f} |",
            f"val subject AP: {val_history.subject_average_precision[-1]:.4f}",
        )

    if save:
        plots = [
            ("loss", "Loss"),
            ("balanced_accuracy", "Balanced Accuracy"),
            ("roc_auc", "ROC-AUC"),
            ("average_precision", "Average Precision"),
            ("subject_balanced_accuracy", "Subject Balanced Accuracy"),
            ("subject_roc_auc", "Subject ROC-AUC"),
            ("subject_average_precision", "Subject Average Precision"),
        ]
        for attr, name in plots:
            plot_metric(
                getattr(train_history, attr),
                getattr(val_history, attr),
                name,
                save_path=output_dir / f"fold_{fold}_{attr}.png",
            )


def validate_model(model, val_loader, criterion, use_amp=False):
    model.eval()
    device = next(model.parameters()).device

    running_loss = 0.0
    total = 0
    all_logits = []
    all_labels = []
    all_subjects = []

    for batch in val_loader:
        inputs = batch["image"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)

        with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        total += outputs.size(0)
        all_logits.append(outputs.float().cpu())
        all_labels.append(labels.cpu())
        all_subjects.extend(batch["subject"])

    return running_loss / total, *compute_metrics(
        torch.cat(all_labels), torch.cat(all_logits), all_subjects
    )

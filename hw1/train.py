import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from model import Network
from utils import make_dataset, seed_everything, show_curves, weights_init


class ArgumentParser(Tap):
    data_folder: str = "data"  # the root of data folder
    output_folder: str = "results"
    lr: float = 0.001  # the learning rate for training
    momentum: float = 0.9
    weight_decay: float = 0
    batch_size: int = 100  # the size of mini-batch
    epoch_size: int = 100  # the size of epochs
    every_num_epochs_for_val: int = 10
    num_workers: int = 0
    seed: int = 1234


def train(
    model: Network,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> dict:
    model.train()

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    batch_metrics = dict(train_loss=[], train_acc=[])
    for imgs, labels in tqdm(dataloader, desc="Train batch", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        pred_logits: torch.Tensor = model(imgs)

        loss = loss_fn(pred_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_labels = F.softmax(pred_logits.detach(), dim=1)
        acc = torch.eq(torch.argmax(pred_labels, dim=1), labels).float().mean()
        batch_metrics["train_loss"].append(loss.item())
        batch_metrics["train_acc"].append(acc.item())

    return batch_metrics


@torch.no_grad()
def evaluate(model: Network, dataloader: DataLoader, device: str = "cpu"):
    model.eval()

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    batch_metrics = dict(val_loss=[], val_acc=[])
    for imgs, labels in tqdm(dataloader, desc="Val batch", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        pred_logits: torch.Tensor = model(imgs)

        loss = loss_fn(pred_logits, labels)
        pred_labels = F.softmax(pred_logits, dim=1)
        acc = torch.eq(torch.argmax(pred_labels, dim=1), labels).float().mean()

        batch_metrics["val_loss"].append(loss.item())
        batch_metrics["val_acc"].append(acc.item())

    return batch_metrics


def main(args: ArgumentParser):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    train_dataset = make_dataset(args.data_folder, "train")
    val_dataset = make_dataset(args.data_folder, "val")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Network(3, 10)
    model.apply(weights_init)
    logging.info(f"# of parameters in model: {model.num_parameters}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.momentum, 0.999),
        weight_decay=args.weight_decay,
    )

    model.to(device)
    metrics = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    best_val_acc = 0
    for epoch in trange(1, args.epoch_size + 1, desc="Epoch"):
        batch_metrics = train(model, train_dataloader, optimizer, device)
        for key in batch_metrics.keys():
            metrics[key].append(np.mean(batch_metrics[key]))

        logging.info("#######################################################")
        metric_msg = f"""
            Train loss: {metrics['train_loss'][-1]}\t
            Train acc: {metrics['train_acc'][-1]}
        """
        logging.info(f"Epoch {epoch}/{args.epoch_size}\t{metric_msg}")

        if epoch % args.every_num_epochs_for_val == 0:
            batch_metrics = evaluate(model, val_dataloader, device)
            for key in batch_metrics.keys():
                metrics[key].append(np.mean(batch_metrics[key]))

            if best_val_acc < metrics["val_acc"][-1]:
                best_val_acc = metrics["val_acc"][-1]
                content = dict(model=model.state_dict(), metrics=metrics)
                torch.save(
                    content,
                    Path(args.output_folder) / f"best_model_epoch_{epoch}.pth",
                )

            metric_msg = f"""
                Val loss: {metrics['val_loss'][-1]}\t
                Val acc: {metrics['val_acc'][-1]}
            """
            logging.info(f"Epoch {epoch}/{args.epoch_size}\t{metric_msg}")

    show_curves(args.output_folder, metrics, args.every_num_epochs_for_val)


if __name__ == "__main__":
    args = ArgumentParser().parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    args.output_folder = str(
        Path(args.output_folder) / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    os.makedirs(args.output_folder, exist_ok=True)

    logging.basicConfig(
        filename=Path(args.output_folder) / "info.log",
        format="%(levelname)s:%(message)s",
        level=logging.INFO,
    )

    logging.info(args)
    main(args)

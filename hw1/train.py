from datetime import datetime
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from model import Network
from utils import make_dataset, seed_everything


class ArgumentParser(Tap):
    data_folder: str = "data"  # the root of data folder
    output_folder: str = "results"
    lr: float = 0.001  # the learning rate for training
    momentum: float = 0.9
    batch_size: int = 100  # the size of mini-batch
    epoch_size: int = 100  # the size of epochs
    every_num_epochs_for_val: int = 10
    num_workers: int = 0
    seed: int = 1234


def train(
    model: Network,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> dict:
    model.train()

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    batch_metrics = dict(train_loss=[], train_acc=[])
    for imgs, labels in tqdm(dataloader, desc="Batch", leave=False):
        pred_labels: torch.Tensor = model(imgs)

        loss = loss_fn(pred_labels, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = torch.eq(torch.argmax(pred_labels, dim=1), labels).sum().float()
        batch_metrics["train_loss"].append(loss.item())
        batch_metrics["train_acc"].append(acc.item())

    return batch_metrics


def evaluate(model: Network, dataloader: DataLoader):
    model.eval()

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    batch_metrics = dict(val_loss=[], val_acc=[])
    for imgs, labels in tqdm(dataloader, desc="Batch", leave=False):
        pred_labels: torch.Tensor = model(imgs)

        loss = loss_fn(pred_labels, labels)
        acc = torch.eq(torch.argmax(pred_labels, dim=1), labels).sum().float()

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
    logging.info(f"# of parameters in model: {model.num_parameters}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(args.momentum, 0.999)
    )

    model.to(device)
    metrics = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    best_val_acc = 0
    for epoch in trange(args.epoch_size, desc="Epoch"):
        batch_metrics = train(model, train_dataloader, optimizer)

        for key in batch_metrics.keys():
            metrics[key].append(np.mean(batch_metrics[key]))

        metric_msg = f"""
            Train loss: {metrics['train_loss'][-1]}\t
            Train acc: {metrics['train_acc'][-1]}
        """
        logging.info(f"Epoch {epoch}/{args.epoch_size}\t{metric_msg}")

        if epoch % args.every_num_epochs_for_val == 0:
            logging.info("Validating model...")
            val_metrics = evaluate(model, val_dataloader)
            for key in val_metrics.keys():
                metrics[key].append(np.mean(val_metrics[key]))

            if best_val_acc < metrics["val_acc"][-1]:
                best_val_acc = metrics["val_acc"][-1]
                content = dict(model=model.state_dict(), metrics=metrics)
                torch.save(content, Path(args.output_folder) / "model.pth")

            metric_msg = f"""
                Val loss: {metrics['val_loss'][-1]}\t
                Val acc: {metrics['val_acc'][-1]}
            """
            logging.info(f"Epoch {epoch}/{args.epoch_size}\t{metric_msg}")

    # TODO: Generate loss-epoch & acc-epoch curve graph for train and val


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

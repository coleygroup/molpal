from typing import Dict

from ray import train
import ray.train.torch  # noqa: F401
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from molpal.models import mpnn
from molpal.models.chemprop.data.data import construct_molecule_batch
from molpal.models.chemprop.nn_utils import NoamLR


def train_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    uncertainty: str = "none",
):
    model.train()

    Ls = []
    num_samples = 0

    for i, batch in enumerate(train_loader):
        # batch_info = {"batch_idx": i}
        Xs, Y = batch

        mask = ~torch.isnan(Y)
        Y = torch.nan_to_num(Y, nan=0.0)
        class_weights = torch.ones_like(Y)

        Y_pred = model(Xs)

        if uncertainty == "mve":
            Y_pred_mean = Y_pred[:, 0::2]
            Y_pred_var = Y_pred[:, 1::2]
            L = criterion(Y_pred_mean, Y_pred_var, Y)
        else:
            L = criterion(Y_pred, Y) * class_weights * mask

        L = L.sum() / mask.sum()

        optimizer.zero_grad()
        L.backward()

        optimizer.step()
        scheduler.step()

        Ls.append(L)
        num_samples += len(Y)

    L = torch.stack(Ls).mean().item()

    return {"loss": L, "num_samples": num_samples}


@torch.no_grad()
def validate_epoch(
    val_loader: DataLoader, model: nn.Module, metric: nn.Module, uncertainty: str = "none"
):
    model.eval()

    Ls = []
    num_samples = 0

    for i, batch in enumerate(val_loader):
        # batch_info = {"batch_idx": batch_idx}
        # step_results = validate_step(batch, model, metric, device, uncertainty)
        # losses.append(step_results["loss"])
        # num_samples += step_results["num_samples"]
        Xs, Y = batch

        Y_pred = model(Xs)
        if uncertainty == "mve":
            Y_pred = Y_pred[:, 0::2]

        Ls.append(metric(Y_pred, Y))
        num_samples += len(Y)

    L = torch.cat(Ls).mean().item()

    return {"loss": L, "num_samples": num_samples}


def train_func(config: Dict):
    model = config["model"]
    train_data = config["train_data"]
    val_data = config["val_data"]
    uncertainty = config["uncertainty"]
    dataset_type = config["dataset_type"]
    metric = config.get("metric", "rmse")
    batch_size = config.get("batch_size", 50)
    warmup_epochs = config.get("warmup_epochs", 2.0)
    max_epochs = config.get("max_epochs", 50)
    num_lrs = 1
    init_lr = config.get("init_lr", 1e-4)
    max_lr = config.get("max_lr", 1e-3)
    final_lr = config.get("final_lr", 1e-4)
    ncpu = config.get("ncpu", 1)

    train_loader = DataLoader(
        train_data,
        batch_size,
        # sampler=DistributedSampler(train_data),
        num_workers=ncpu,
        collate_fn=construct_molecule_batch,
    )
    val_loader = DataLoader(
        val_data,
        batch_size,
        # sampler=DistributedSampler(val_data),
        num_workers=ncpu,
        collate_fn=construct_molecule_batch,
    )

    optimizer = Adam(model.parameters(), init_lr, weight_decay=0)
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[warmup_epochs],
        total_epochs=[max_epochs] * num_lrs,
        steps_per_epoch=len(train_data) / batch_size + 1,
        init_lr=[init_lr],
        max_lr=[max_lr],
        final_lr=[final_lr],
    )
    criterion = mpnn.utils.get_loss_func(dataset_type, uncertainty)
    metric = {
        "mse": lambda X, Y: F.mse_loss(X, Y, reduction="none"),
        "rmse": lambda X, Y: torch.sqrt(F.mse_loss(X, Y, reduction="none")),
    }[metric]

    model = train.torch.prepare_model(model)
    train_loader = train.torch.prepare_data_loader(train_loader)
    val_loader = train.torch.prepare_data_loader(val_loader)

    for i in range(max_epochs):
        train_res = train_epoch(train_loader, model, criterion, optimizer, scheduler, uncertainty)
        val_res = validate_epoch(val_loader, model, metric, uncertainty)

        train_loss = train_res["loss"]
        val_loss = val_res["loss"]

        train.report(epoch=i, train_loss=train_loss, val_loss=val_loss)

    return model.module.to("cpu")

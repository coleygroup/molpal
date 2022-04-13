from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from molpal.models.chemprop.data import MoleculeDataLoader
from molpal.models.chemprop.nn_utils import NoamLR


def train(
    model: nn.Module,
    data_loader: MoleculeDataLoader,
    loss_func: Callable,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    uncertainty: bool,
    n_iter: int = 0,
    disable: bool = False,
) -> int:
    """Trains a model for an epoch

    Parameters
    ----------
    model : nn.Module
        the model to train
    data_loader : MoleculeDataLoader
        an iterable of MoleculeDatasets
    loss_func : Callable
        the loss function
    optimizer : Optimizer
        the optimizer
    scheduler : _LRScheduler
        the learning rate scheduler
    uncertainty : bool
        whether the model predicts its own uncertainty
    n_iter : int, default=0
        the current number of training iterations
    disable : bool, default=False
        whether to disable the progress bar

    Returns
    -------
    n_iter : int
        The total number of samples trained on so far
    """
    model.train()
    # loss_sum = 0
    # iter_count = 0

    for batch in tqdm(data_loader, desc="Training", unit="step", leave=False, disable=disable):
        mol_batch, targets = batch

        model.zero_grad()
        preds = model(mol_batch)  # , features_batch)

        mask = torch.tensor([list(map(bool, ys)) for ys in targets]).to(preds.device)
        targets = torch.tensor([[y or 0 for y in ys] for ys in targets]).to(preds.device)
        class_weights = torch.ones(targets.shape).to(preds.device)

        # if args.dataset_type == 'multiclass':
        #     targets = targets.long()
        #     loss = (torch.cat([
        #         loss_func(preds[:, target_index, :],
        #                    targets[:, target_index]).unsqueeze(1)
        #         for target_index in range(preds.size(1))
        #         ], dim=1) * class_weights * mask
        #     )

        if uncertainty:
            pred_means = preds[:, 0::2]
            pred_vars = preds[:, 1::2]

            loss = loss_func(pred_means, pred_vars, targets)
        else:
            loss = loss_func(preds, targets) * class_weights * mask

        loss = loss.sum() / mask.sum()

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

    return n_iter

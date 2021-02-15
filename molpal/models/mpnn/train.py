from argparse import Namespace
from logging import Logger
from typing import Callable, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from ..chemprop.data import MoleculeDataLoader
from ..chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR

def train(model: nn.Module, data_loader: MoleculeDataLoader,
          loss_func: Callable, optimizer: Optimizer,
          scheduler: _LRScheduler, args: Namespace,
          n_iter: int = 0, logger: Optional[Logger] = None,
          writer: Optional = None) -> int:
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
    args : Namespace
        a Namespace object the containing necessary attributes
    n_iter : int
        the current number of training iterations
    logger : Optional[Logger] (Default = None)
        a logger for printing intermediate results. If None, print
        intermediate results to stdout
    writer : Optional[SummaryWriter] (Default = None)
        A tensorboardX SummaryWriter

    Returns
    -------
    n_iter : int
        The total number of iterations (training examples) trained on so far
    """
    model.train()
    # loss_sum = 0
    # iter_count = 0

    for batch in tqdm(data_loader, desc='Step', unit='minibatch',
                      leave=False):
        # Prepare batch
        mol_batch = batch.batch_graph()
        features_batch = batch.features()

        # Run model
        model.zero_grad()
        preds = model(mol_batch, features_batch)        

        targets = batch.targets()   # targets might have None's
        mask = torch.Tensor(
            [list(map(bool, ys)) for ys in targets]).to(preds.device)
        targets = torch.Tensor(
            [[y or 0 for y in ys] for ys in targets]).to(preds.device)
        class_weights = torch.ones(targets.shape).to(preds.device)
        
        # if args.dataset_type == 'multiclass':
        #     targets = targets.long()
        #     loss = (torch.cat([
        #         loss_func(preds[:, target_index, :],
        #                    targets[:, target_index]).unsqueeze(1)
        #         for target_index in range(preds.size(1))
        #         ], dim=1) * class_weights * mask
        #     )

        if model.uncertainty:
            pred_means = preds[:, 0::2]
            pred_vars = preds[:, 1::2]

            loss = loss_func(pred_means, pred_vars, targets)
        else:
            loss = loss_func(preds, targets) * class_weights * mask

        loss = loss.sum() / mask.sum()

        # loss_sum += loss.item()
        # iter_count += len(batch)

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        # if (n_iter // args.batch_size) % args.log_frequency == 0:
        #     lrs = scheduler.get_lr()
        #     pnorm = compute_pnorm(model)
        #     gnorm = compute_gnorm(model)
        #     loss_avg = loss_sum / iter_count
        #     loss_sum, iter_count = 0, 0

        #     lrs_str = ', '.join(
        #         f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
        #     debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, '
        #           + f'GNorm = {gnorm:.4f}, {lrs_str}')

        #     if writer:
        #         writer.add_scalar('train_loss', loss_avg, n_iter)
        #         writer.add_scalar('param_norm', pnorm, n_iter)
        #         writer.add_scalar('gradient_norm', gnorm, n_iter)
        #         for i, lr in enumerate(lrs):
        #             writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter

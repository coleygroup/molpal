from typing import List, Iterable, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ..chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from ..chemprop.features import BatchMolGraph, mol2graph

def predict(model: nn.Module, data_loader: Iterable,
            disable_progress_bar: bool = False,
            scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """Predict the output values of a dataset

    Parameters
    ----------
    model : nn.Module
        the model to use
    data_loader : MoleculeDataLoader
        an iterable of MoleculeDatasets
    disable_progress_bar : bool (Default = False)
        whether to disable the progress bar
    scaler : Optional[StandardScaler] (Default = None)
        A StandardScaler object fit on the training targets

    Returns
    -------
    predictions : np.ndarray
        an NxM array where N is the number of inputs for wchi to produce 
        predictions and M is the number of prediction tasks
    """
    model.eval()

    pred_batches = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Batch inference', unit='minibatch',
                          leave=False):
            batch_graph = batch.batch_graph()
            pred_batch = model(batch_graph)
            pred_batches.append(pred_batch.data.cpu().numpy())
    preds = np.concatenate(pred_batches)

    if model.uncertainty:
        means = preds[:, 0::2]
        variances = preds[:, 1::2]

        if scaler:
            means = scaler.inverse_transform(means)
            variances = scaler.stds**2 * variances

        return means, variances

    # Inverse scale if regression
    if scaler:
        preds = scaler.inverse_transform(preds)

    return preds

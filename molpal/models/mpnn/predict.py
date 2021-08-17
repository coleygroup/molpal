from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ..chemprop.data import (
    StandardScaler, MoleculeDataLoader, MoleculeDataset, MoleculeDatapoint
)

def predict(model, smis: Iterable[str], batch_size: int = 50, ncpu: int = 1, 
            uncertainty: Optional[str] = None,
            scaler: Optional[StandardScaler] = None,
            use_gpu: bool = False, disable: bool = False):
    """Predict the target values of the given SMILES strings with the
    input model

    Parameters
    ----------
    model : mpnn.MoleculeModel
        the model to use
    smis : Iterable[str]
        the SMILES strings to perform inference on
    batch_size : int, default=50
        the size of each minibatch (impacts performance)
    ncpu : int, default=1
        the number of cores over which to parallelize input preparation
    uncertainty : Optional[str], default=None
        the method the model uses for uncertainty quantification. None, if the
        model does not inherently quantify uncertainty
    scaler : StandardScaler, default=None
        A StandardScaler object fit on the training targets. If none,
        prediction values will not be transformed to original dataset
    use_gpu : bool, default=False
        whether to use the GPU during inference
    disable : bool, default=False
        whether to disable the progress bar

    Returns
    -------
    preds : np.ndarray
        an array containing the predictions for each input SMILES string. Array
        is of shape (N, M), where N is number of SMILES strings and M is the
        number of tasks to predict for. If only single-task prediction, then
        array will be be of shape (N,). If predicting uncertainty, array is of
        shape (N, 2M), where the mean predicted values are indices [:, 0::2] and
        the predicted variances are indices [:, 1::2]
    """
    device = 'cuda' if use_gpu else 'cpu'
    model.to(device)

    dataset = MoleculeDataset([MoleculeDatapoint([smi]) for smi in smis])
    data_loader = MoleculeDataLoader(
        dataset=dataset, batch_size=batch_size,
        num_workers=ncpu, pin_memory=use_gpu
    )
    model.eval()

    pred_batches = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Inference', unit='batch',
                          leave=False, disable=disable):
            componentss, _ = batch
            componentss = [
                [X.to(device)#, non_blocking=True)
                 if isinstance(X, torch.Tensor) else X for X in components]
                for components in componentss
            ]
            pred_batch = model(componentss)
            pred_batches.append(pred_batch)#.data.cpu().numpy())

        preds = torch.cat(pred_batches)
    preds = preds.cpu().numpy()

    if uncertainty in ('mve', 'evidential'):
        if uncertainty == 'evidential':
            means = preds[:, 0::4]
            lambdas = preds[:, 1::4]
            alphas = preds[:, 2::4]
            betas = preds[:, 3::4]

            # NOTE: inverse-evidence (ie. 1/evidence) is also a measure of
            # confidence. we can use this or the Var[X] defined by NIG.
            inverse_evidences = 1. / ((alphas-1) * lambdas)
            variances = inverse_evidences * betas
            
            preds = np.empty((len(preds), 2))
            preds[:, 0::2] = means
            preds[:, 1::2] = variances

        if scaler:
            preds[:, 0::2] *= scaler.stds
            preds[:, 0::2] += scaler.means
            preds[:, 1::2] *= scaler.stds**2

        return preds

    if scaler:
        preds *= scaler.stds
        preds += scaler.means
        # preds = scaler.inverse_transform(preds)

    return preds

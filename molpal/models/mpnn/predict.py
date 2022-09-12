from typing import Iterable, Optional

import numpy as np
import ray
import torch
from tqdm import tqdm

from molpal.models.mpnn.model import MoleculeModel
from molpal.models.chemprop.data import (
    StandardScaler,
    MoleculeDataLoader,
    MoleculeDataset,
    MoleculeDatapoint,
)


@torch.inference_mode()
def predict(
    model: MoleculeModel,
    smis: Iterable[str],
    batch_size: int = 50,
    ncpu: int = 1,
    uncertainty: Optional[str] = None,
    scaler: Optional[StandardScaler] = None,
    use_gpu: bool = False,
    disable: bool = False,
) -> np.ndarray:
    """Predict the target values of the given SMILES strings with the
    input model

    Parameters
    ----------
    model : MoleculeModel
        the model to use
    smis : Iterable[str]
        the SMILES strings of the molecules to predict properties for
    batch_size : int, default=50
        the size of each minibatch
    ncpu : int, default=1
        the number of cores over which to parallelize input preparation
    uncertainty : Optional[str], default=None
        the uncertainty quantifiacation method the model uses. None if it
        does not use any uncertainty quantifiacation
    scaler : StandardScaler, default=None
        A StandardScaler object fit on the training targets. If none,
        prediction values will not be transformed to original dataset
    use_gpu : bool, default=False
        whether to use the GPU during inference
    disable : bool, default=False
        whether to disable the progress bar

    Returns
    -------
    Y_pred : np.ndarray
        an `n x m` array where `n` is the number of SMILES strings and `m` is the number of tasks
    """
    model.eval()

    device = "cuda" if use_gpu else "cpu"
    model.to(device)

    dataset = MoleculeDataset([MoleculeDatapoint([smi]) for smi in smis])
    data_loader = MoleculeDataLoader(dataset, batch_size, ncpu)

    Y_pred_batches = []
    for batch in tqdm(data_loader, "Inference", unit="batch", leave=False, disable=disable):
        componentss, _ = batch
        componentss = [
            [X.to(device) if torch.is_tensor(X) else X for X in components]
            for components in componentss
        ]
        Y_pred_batches.append(model(componentss))

    Y_pred = torch.cat(Y_pred_batches)
    Y_pred = Y_pred.cpu().numpy()

    # if uncertainty == "mve":
    #     if scaler:
    #         Y_pred[:, 0::2] *= scaler.stds
    #         Y_pred[:, 0::2] += scaler.means
    #         Y_pred[:, 1::2] *= scaler.stds**2

    #     return Y_pred

    # if scaler:
    #     Y_pred *= scaler.stds
    #     Y_pred += scaler.means

    return Y_pred


@ray.remote
def predict_(*args, **kwargs):
    return predict(*args, **kwargs)

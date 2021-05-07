from typing import Optional

import numpy as np
import torch
from torch import clamp, log, nn

def get_loss_func(dataset_type: str,
                  uncertainty: Optional[str] = None) -> nn.Module:
    """Get the loss function corresponding to a given dataset type

    Parameters
    ----------
    dataset_type : str
        the type of dataset
    uncertainty : Optional[str]
        the method the model uses for uncertainty quantification. None if the
        model does not quantify its uncertainty.

    Returns
    -------
    loss_function : nn.Module
        a PyTorch loss function

    Raises
    ------
    ValueError
        if is dataset_type is neither "classification" nor "regression"
    """
    if dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')

    elif dataset_type == 'regression':
        if uncertainty == 'mve':
            return mve_loss
        elif uncertainty == 'evidential':
            return evidential_loss

        return nn.MSELoss(reduction='none')

    raise ValueError(f'Unsupported dataset type: "{dataset_type}."')

def mve_loss(pred_means, pred_vars, targets):
    """The NLL loss function as defined in:
    Nix, D.; Weigend, A. ICNN’94. 1994; pp 55–60 vol.1"""
    clamped_var = torch.clamp(pred_vars, min=0.00001)
    return (
        torch.log(clamped_var)/2
        + (pred_means - targets)**2 / (2*clamped_var)
    )

def evidential_loss(means, lambdas, alphas, betas, targets,
                    lam=1, epsilon=1e-4):
    """Use Deep Evidential Regression negative log likelihood loss + evidential
    regularizer

    Parameters
    -----------
    mu
        pred mean parameter for NIG
    :v:
        pred lam parameter for NIG
    :alpha:
        predicted parameter for NIG
    :beta:
        Predicted parmaeter for NIG
    :targets:
        Outputs to predict
    :return: Loss
    """
    twoBlambda = 2 * betas*(1 + lambdas)
    nll = (
        0.5 * torch.log(np.pi / lambdas)
        - alphas * torch.log(twoBlambda)
        + (alphas + 0.5) * torch.log(lambdas*(targets - means)**2 + twoBlambda)
        + torch.lgamma(alphas)
        - torch.lgamma(alphas + 0.5)
    )

    # L_NLL = nll #torch.mean(nll, dim=-1)

    # Calculate regularizer based on absolute error of prediction
    error = torch.abs((targets - means))
    reg = error * (2 * lambdas + alphas)
    # L_REG = reg #torch.mean(reg, dim=-1)

    # Loss = L_NLL + L_REG
    # TODO If we want to optimize the dual- of the objective use the line below:
    loss = nll + lam * (reg - epsilon)

    return loss

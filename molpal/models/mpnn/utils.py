from typing import Optional

import numpy as np
import torch
from torch import nn

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
                    c: float = 0., epsilon=1e-4) -> torch.Tensor:
    """Use Deep Evidential Regression negative log likelihood loss + evidential
    regularizer

    Parameters
    -----------
    means
        pred mean parameters for NIG
    lambdas
        pred lamda parameters for NIG
    alphas
        predicted alphas parameters for NIG
    betas
        predicted beta parameters for NIG
    targets
        true target means
    c : float, default=0.
        the regularization coefficient
    Returns
    -------
    torch.Tensor
        the loss
    """
    twoBlambda = 2*betas*(1 + lambdas)
    L_nll = (
        0.5 * torch.log(np.pi / lambdas)
        - alphas * torch.log(twoBlambda)
        + (alphas + 0.5) * torch.log(lambdas*(targets - means)**2 + twoBlambda)
        + torch.lgamma(alphas)
        - torch.lgamma(alphas + 0.5)
    )

    L_reg = torch.abs((targets - means)) * (2*lambdas + alphas)

    loss = L_nll + c*(L_reg - epsilon)

    return loss

def gamma(x):
    return torch.exp(torch.lgamma(x))

def evidential_loss_(means, lambdas, alphas, betas, targets) -> torch.Tensor:
    """Use Deep Evidential Regression Sum of Squared Error loss

    Parameters
    -----------
    means
        pred mean parameter for NIG
    lambdas
        pred lam parameter for NIG
    alphas
        predicted parameter for NIG
    betas
        Predicted parmaeter for NIG
    targets
        Outputs to predict
    
    Returns
    -------
    torch.Tensor
        the loss
    """
    coeff_denom = 4 * gamma(alphas) * lambdas * torch.sqrt(betas)
    coeff_num = gamma(alphas - 0.5)
    coeff = coeff_num / coeff_denom

    second_term = 2 * betas * (1 + lambdas)
    second_term += (2 * alphas - 1) * lambdas * torch.pow((targets - means), 2)

    L_sos = coeff * second_term
    L_reg = (targets - means)**2 * (2*alphas + lambdas)

    loss_val = L_sos + L_reg

    return loss_val

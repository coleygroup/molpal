from typing import Optional

from torch import clamp, log, nn

def get_loss_func(dataset_type: str,
                  uncertainty_method: Optional[str] = None) -> nn.Module:
    """Get the loss function corresponding to a given dataset type

    Parameters
    ----------
    dataset_type : str
        the type of dataset
    uncertainty_method : Optional[str]
        the uncertainty method being used

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
        if uncertainty_method == 'mve':
            return negative_log_likelihood

        return nn.MSELoss(reduction='none')

    raise ValueError(f'Unsupported dataset type: "{dataset_type}."')

def negative_log_likelihood(pred_mean, pred_var, targets):
    """The NLL loss function as defined in:
    Nix, D.; Weigend, A. ICNN’94. 1994; pp 55–60 vol.1"""
    clamped_var = clamp(pred_var, min=0.00001)
    return (
        log(clamped_var)/2
        + (pred_mean - targets)**2 / (2*clamped_var)
    )


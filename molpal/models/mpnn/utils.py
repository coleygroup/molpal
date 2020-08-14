from concurrent.futures import ProcessPoolExecutor as Pool
from typing import Iterable, Iterator, Optional

from torch import clamp, log, nn

from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.features import BatchMolGraph, mol2graph

from ..utils import batches

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
    return (log(clamped_var)/2
            + (pred_mean - targets)**2/(2*clamped_var))

def batch_graphs(smis: Iterable[str], minibatch_size: int = 50,
                 n_workers: int = 1) -> Iterator[BatchMolGraph]:
    """Generate BatchMolGraphs from the SMILES strings

    Uses parallel processing to buffer a chunk of BatchMolGraphs into memory,
    where the chunksize is equal to the number of workers available. Only
    prepares one chunk at a time due to the exceedingly large memory footprint 
    of a BatchMolGraph

    Parameters
    ----------
    smis : Iterable[str]
        the SMILES strings from which to generate BatchMolGraphs
    minibatch_size : int
        the number of molecular graphs in each BatchMolGraph
    n_workers : int
        the number of workers to parallelize BatchMolGraph preparation over
    
    Yields
    ------
    BatchMolGraph
        a batch of molecular graphs of size <minibatch_size>
    """
    # need a dataset if we're going to use features
    # test_data = MoleculeDataset([
    #     MoleculeDatapoint(smiles=smi,) for smi in smis
    # ])
    chunksize = minibatch_size*n_workers
    with Pool(max_workers=n_workers) as pool:
        for chunk_smis in batches(smis, chunksize):
            smis_minibatches = list(batches(chunk_smis, minibatch_size))
            for batch_graph in pool.map(mol2graph, smis_minibatches):
                yield batch_graph


import csv
from functools import partial
import gzip
from pathlib import Path
from typing import Collection, Dict, Iterable, Optional

import numpy as np
from tqdm import tqdm

from molpal.objectives.base import Objective

class LookupObjective(Objective):
    """A LookupObjective calculates the objective function by looking the
    value up in an input file.

    Useful for retrospective studies.

    Attributes
    ----------
    self.data : str
        the path of a file containing a Shelf object that holds a dictionary 
        mapping an input string to its objective function value

    Parameters
    ----------
    lookup_path : str
        the path of the file containing lookup data
    lookup_title_line : bool (Default = True)
        is there a title in in the lookup file?
    lookup_smiles_col : int (Default = 0)
        the column containing the SMILES string in the lookup file
    lookup_data_col : int (Default = 1)
        the column containing the desired data in the lookup file
    **kwargs
        unused and addditional keyword arguments
    """
    def __init__(self, lookup_path: str,
                 lookup_sep: str = ',', lookup_title_line: bool = True,
                 lookup_smiles_col: int = 0, lookup_data_col: int = 1,
                 **kwargs):
        if Path(lookup_path).suffix == '.gz':
            open_ = partial(gzip.open, mode='rt')
        else:
            open_ = open
        
        self.data = {}
        with open_(lookup_path) as fid:
            reader = csv.reader(fid, delimiter=lookup_sep)
            if lookup_title_line:
                next(fid)

            for row in tqdm(reader, desc='Building oracle'):
                # assume all data is a float value right now
                key = row[lookup_smiles_col]
                val = row[lookup_data_col]
                try:
                    self.data[key] = float(val)
                except ValueError:
                    pass

        super().__init__(**kwargs)

    def calc(self, smis: Collection[str],
             *args, **kwargs) -> Dict[str, Optional[float]]:
        return {
            smi: self.c * self.data[smi] if smi in self.data else None
            for smi in smis
        }
    
    def residuals(self, smis: Iterable[str], Y_pred: np.ndarray) -> np.ndarray:
        """
        return the residuals of the predictions

        Parameters
        ----------
        smis : Iterable[str]
            the SMILES strings corresponding to each prediction
        Y_pred : np.ndarr
            the predicted means

        Returns
        -------
        np.ndarray
            the residuals for each prediction. SMILES strings with no 
            corresponding objective function value will have a residual of 0
        """
        Y_true = np.array([self.data.get(smi) for smi in smis], dtype=float)
        mask = np.isnan(Y_true)

        Y_true[mask] = 0
        R = Y_true - Y_pred
        R[mask] = 0
        return R
        
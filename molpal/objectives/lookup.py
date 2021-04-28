import csv
from functools import partial
import gzip
from pathlib import Path
from typing import Collection, Dict, Optional

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
    
    def residuals(self, smis, Y_hat: np.ndarray) -> np.ndarray:
        Y = np.array([self.data.get(smi, 0.) for smi in smis])
        # mask = ~np.isnan(Y)

        # Y = self.c * Y[mask]
        # Y_hat = Y_hat[mask]

        return np.abs(Y - Y_hat)
        
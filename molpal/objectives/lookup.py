import csv
from functools import partial
import gzip
from pathlib import Path
from typing import Collection, Dict, Optional

from configargparse import ArgumentParser
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
    def __init__(self, objective_config: str,
                #  lookup_path: str,
                #  lookup_sep: str = ',', lookup_title_line: bool = True,
                #  lookup_smiles_col: int = 0, lookup_data_col: int = 1,
                 **kwargs):
        path, sep, title_line, smiles_col, score_col, minimize = (
            parse_config(objective_config))

        if Path(path).suffix == '.gz':
            open_ = partial(gzip.open, mode='rt')
        else:
            open_ = open
        
        self.data = {}
        with open_(path) as fid:
            reader = csv.reader(fid, delimiter=sep)
            if title_line:
                next(fid)

            for row in tqdm(reader, desc='Building oracle'):
                # assume all data is a float value right now
                key = row[smiles_col]
                val = row[score_col]
                try:
                    self.data[key] = float(val)
                except ValueError:
                    pass

        super().__init__(minimize=minimize)

    def calc(self, smis: Collection[str],
             *args, **kwargs) -> Dict[str, Optional[float]]:
        return {
            smi: self.c * self.data[smi] if smi in self.data else None
            for smi in smis
        }
        
def parse_config(config: str):
    parser = ArgumentParser()
    parser.add_argument('config', is_config_file=True)
    parser.add_argument('--path', required=True)
    parser.add_argument('--sep', default=',')
    parser.add_argument('--no-title-line', action='store_true', default=False)
    parser.add_argument('--smiles-col', type=int, default=0)
    parser.add_argument('--score-col', type=int, default=1)
    parser.add_argument('--minimize', action='store_true', default=False)

    params = vars(parser.parse_args(config))
    return (
        params['path'], params['sep'], not params['no_title_line'],
        params['smiles_col'], params['score_col'], params['minimize']
    )
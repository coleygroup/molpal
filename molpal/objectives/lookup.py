import csv
from functools import partial
import gzip
from pathlib import Path
from typing import Collection, Dict, Optional

from configargparse import ArgumentParser
from tqdm import tqdm

from molpal.objectives.base import Task

class LookupTask(Task):
    """A LookupTask calculates the objective function by looking the
    value up in an input file.

    Useful for retrospective studies.

    Attributes
    ----------
    self.data : str
        the path of a file containing a Shelf object that holds a dictionary 
        mapping an input string to its objective function value

    Parameters
    ----------
    objective_config : str
        the path to a pyscreener config file containing the options for
        docking calculations.
    verbose : int, default=0
    minimize : bool, default=True
    """
    def __init__(self, config: str, *,
                 path: str, minimize: bool = True, verbose: int = 0
                ):
        path, sep, title_line, smiles_col, score_col, minimize = (
            parse_config(config)
        )

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
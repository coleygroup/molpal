import atexit
import csv
from typing import Dict, List, Optional, TypeVar

from molpal.objectives.base import Objective

from pyscreener import args as pyscreener_args
from pyscreener import docking

T = TypeVar('T')
U = TypeVar('U')

class DockingObjective(Objective):
    """A DockingObjective calculates the objective function by calculating the
    docking score of a molecule

    Attributes
    ----------
    c : int
        the min/maximization constant, depending on the objective
    docking_screener : Pyscreener.docking.Screener
        the pyscreener screening object that handles python-based calls to
        docking software. The screener is an object that holds the
        information of a receptor active site and docks the ligands 
        corresponding to input SMILES strings into that active site

    Parameters
    ----------
    config : Optional[str] (Default = None)
        the path to a pyscreener config file containing the options for docking
        calculations.
    verbose : int (Default = 0)
    **kwargs
        additional and unused keyword arguments
    """
    def __init__(self, objective_config: str, path: str = '.',
                #  input_map_file: Optional[str] = None,
                 verbose: int = 0, minimize: bool = True, **kwargs):
        # self.input_map_file = input_map_file
        
        params = {
            'software': 'vina', 'size': (10., 10., 10.),
            'score_mode': 'best', 'ncpu': 1, 'path': path, 'verbose': verbose
        }

        params.update(vars(pyscreener_args.gen_args(
            f'--config {objective_config}'
        )))

        self.docking_screener = docking.screener(**params)

        atexit.register(self.cleanup)
        super().__init__(minimize=minimize)

    def calc(self, smis: List[str],
             **kwargs) -> Dict[str, Optional[float]]:
        """Calculate the docking scores for a list of SMILES strings

        Parameters
        ----------
        smis : List[str]
            a list containing the SMILES strings of ligands to dock
        *args, **kwargs
            additional and unused positional and keyword arguments

        Returns
        -------
        scores : Dict[str, Optional[float]]
            a map from SMILES string to docking score. Ligands that failed
            to dock will be scored as None
        """
        # with shelve.open(self.input_map) as d_smi_inputs:
        #     ligandss = [(smi, d_smi_inputs[smi])
        #                  for smi in smis if smi in d_smi_inputs]
        #     ligands = distribute_and_flatten(ligandss)
        #     extra_smis = [smi for smi in smis if smi not in d_smi_inputs]

        # if extra_smis:
        #     ligands.extend(docking.prepare_ligand(extra_smis, path=in_path))
        scores, full_results = self.docking_screener.dock(
            smis, full_results=True
        )
        self.full_results.extend(full_results)

        return {
            smi: self.c * score if score else None
            for smi, score in scores.items()
        }
    
    def cleanup(self):
        self.docking_screener.collect_files()
        with open(self.docking_screener.path / 'extended.csv', 'w') as fid:
            writer = csv.writer(fid)
            writer.writerow(['smiles', 'name', 'node_id', 'score'])
            writer.writerows(result.values() for result in self.full_results)

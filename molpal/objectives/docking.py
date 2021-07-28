import csv
from functools import partial
import gzip
from pathlib import Path
import shelve
import tempfile
from typing import Dict, List, Optional, Tuple, TypeVar

from molpal.objectives.base import Objective
from molpal.objectives import utils

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
        NOTE: unused right now
    verbose : int (Default = 0)
    **kwargs
        additional and unused keyword arguments
    """
    def __init__(self, objective_config: str,
                 input_map_file: Optional[str] = None,
                 verbose: int = 0, minimize: bool = True, **kwargs):
        # self.input_map_file = input_map_file
        
        params = {
            'software': 'vina', 'size': (10., 10., 10.),
            'score_mode': 'best', 'ncpu': 1, 'path': '.', 'verbose': verbose
        }

        params.update(vars(pyscreener_args.gen_args(
            f'--config {objective_config}'
        )))

        # if software is not None:
        #     docking_params['software'] = software
        # if receptor is not None:
        #     docking_params['receptors'] = [receptor]
        # if box_center is not None:
        #     docking_params['center'] = box_center
        # if box_size is not None:
        #     docking_params['size'] = box_size
        # if docked_ligand_file is not None:
        #     docking_params['docked_ligand_file'] = docked_ligand_file
        # if score_mode is not None:
        #     docking_params['score_mode'] = score_mode
        # if ncpu is not None:
        #     docking_params['ncpu'] = ncpu
        # docking_params['verbose'] = max(docking_params['verbose'], verbose)

        # if name:
        #     path = Path(tempfile.gettempdir()) / name
        # else:
        #     path = Path(tempfile.gettempdir()) / 'molpal_docking'

        self.docking_screener = docking.screener(**params)

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
        scores = self.docking_screener(smis)

        return {
            smi: self.c * score if score else None
            for smi, score in scores.items()
        }

    # def _build_input_map(self, input_map_file) -> str:
    #     """Build the input map dictionary

    #     the input map dictionary is stored on disk using a Shelf object which
    #     is stored in a temporary file.

    #     NOTE: Ideally, the temporary file corresponding to the shelf would only
    #           live for the lifetime of the DockingObjective that owns it.
    #           Unfortunately, there seems no elegant way to do that and, as a 
    #           result, the temporary file will persist until the OS cleans it up

    #     Parameter
    #     ---------
    #     input_map_file : str
    #         a flat csv containing the SMILES string in the 0th column and any
    #         associated input files in the following columns
    #     """
    #     p_input_map_file = Path(input_map_file)
    #     if p_input_map_file.suffix == '.gz':
    #         open_ = partial(gzip.open, mode='rt')
    #     else:
    #         open_ = open

    #     input_map = utils.get_temp_file()

    #     with open_(input_map_file) as fid, \
    #         shelve.open(input_map) as d_smi_inputs:
    #         reader = csv.reader(fid)
            
    #         for smi, *ligands in reader:
    #             d_smi_inputs[smi] = ligands

    #     return input_map

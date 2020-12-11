import csv
from functools import partial
import gzip
from itertools import chain, product
import multiprocessing as mp
import os
from pathlib import Path
import shelve
import tempfile
from typing import Dict, List, Optional, Tuple, TypeVar

from molpal.objectives.base import Objective
from molpal.objectives import utils

from .pyscreener import args as pyscreener_args
from .pyscreener import docking

T = TypeVar('T')
U = TypeVar('U')

class DockingObjective(Objective):
    """A DockingObjective calculates the objective function by calculating the
    docking score of a molecule

    Attributes
    ----------
    c : int = -1
        a DockingObjective is minimized, so c is set to -1
    docking_screener : Pyscreener.docking.Screener
        the pyscreener screening object that handles python-based calls to
        docking software. The screener is an object that holds the
        information of a receptor active site and docks the ligands 
        corresponding to input SMILES strings into that active site

    Parameters
    ----------
    docking_config : Optional[str] (Default = None)
        the path to a pyscreener config file containing the options for docking
        calculations.
        NOTE: specifying any option below in the function call will overwrite
              any values parsed from the configuration file. Absent
              specification in either the config file or the function call, the
              default value will be used.
    software : Optional[str] (Default = 'vina')
        the docking software to use
    receptor : Optional[str] (Default = None)
        the filepath of a receptor file against which to dock the ligands
    center : Optional[Tuple[float, float, float]] (Default = None)
        the center of the docking box
    size : Optional[Tuple[float, float, float]] (Default = (10. 10., 10.))
        the x-, y-, and z-radii of the docking box
    score_mode : Optional[str] (Default = 'best')
        the method to calculate a ligand's overall score from multiple scored
        poses. Choices: 'best' (the best score), 'average' (average all scores),
        and 'boltzmann' (boltzmann average of all scores)
    num_workers : Optional[int] (Default = -1)
        the number of worker processes to spread docking calculations over.
        A value of -1 sets the value equal to the number of cores available.
    ncpu : Optional[int] (Default = 1)
        the number of cores available to each worker processes
    distributed : bool (Default = False)
        whether the worker processes are a distributed setup.
    input_map_file : Optional[str] (Default = None)
        NOTE: unused right now
    verbose : int (Default = 0)
    **kwargs
        additional and unused keyword arguments
    """
    def __init__(self, objective_config: Optional[str] = None,
                 name: Optional[str] = None, root: Optional[str] = None,
                 software: Optional[str] = None,
                 receptor: Optional[str] = None,
                 center: Optional[Tuple] = None,
                 size: Optional[Tuple] = None,
                 score_mode: Optional[str] = None,
                 num_workers: Optional[int] = None,
                 ncpu: Optional[int] = None,
                 distributed: Optional[bool] = None,
                 input_map_file: Optional[str] = None,
                 verbose: int = 0, **kwargs):
        self.input_map_file = input_map_file
        
        docking_params = {
            software: 'vina', size: (10., 10., 10.), score_mode: 'best',
            num_workers: -1, ncpu: 1, distributed: False, verbose: 0
        }
        if objective_config is not None:
            docking_params.update(vars(pyscreener_args.gen_args(
                f'--config {objective_config}'
            )))

        if software is not None:
            docking_params['software'] = software
        if receptor is not None:
            docking_params['receptors'] = [receptor]
        if center is not None:
            docking_params['center'] = center
        if size is not None:
            docking_params['size'] = size
        if score_mode is not None:
            docking_params['score_mode'] = score_mode
        if num_workers is not None:
            docking_params['num_workers'] = num_workers
        if ncpu is not None:
            docking_params['ncpu'] = ncpu
        if distributed is not None:
            docking_params['distributed'] = distributed
        docking_params['verbose'] = max(docking_params['verbose'], verbose)

        if name:
            path = Path(tempfile.gettempdir()) / name
        else:
            path = Path(tempfile.gettempdir()) / 'molpal_docking'

        print(docking_params)
        
        self.docking_screener = docking.screener(**docking_params, path=path)

        exit()
        # if input_map_file:
        #     self.input_map = self._build_input_map(input_map_file)
        # else:
        #     self.input_map = None

        super().__init__(minimize=True)

    def calc(self, smis: List[str],
             in_path: Optional[str] = None,
             out_path: Optional[str] = None,
             **kwargs) -> Dict[str, Optional[float]]:
        """Calculate the docking scores for a list of SMILES strings

        Parameters
        ----------
        smis : List[str]
            a list containing the SMILES strings of ligands to dock
        in_path : Optional[str] (Default = None)
            the path under which docking input files should be written
        out_path : Optional[str] (Default = None)
            the path under which docking output files should be written
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

        if in_path:
            self.docking_screener.in_path = in_path
        if out_path:
            self.docking_screener.out_path = out_path

        scores = self.docking_screener(smis)

        return {
            smi: self.c * score if score else None
            for smi, score in scores.items()
        }

    def _build_input_map(self, input_map_file) -> str:
        """Build the input map dictionary

        the input map dictionary is stored on disk using a Shelf object which
        is stored in a temporary file.

        NOTE: Ideally, the temporary file corresponding to the shelf would only
              live for the lifetime of the DockingObjective that owns it.
              Unfortunately, there seems no elegant way to do that and, as a 
              result, the temporary file will persist until the OS cleans it up

        Parameter
        ---------
        input_map_file : str
            a flat csv containing the SMILES string in the 0th column and any
            associated input files in the following columns
        """
        p_input_map_file = Path(input_map_file)
        if p_input_map_file.suffix == '.gz':
            open_ = partial(gzip.open, mode='rt')
        else:
            open_ = open

        input_map = utils.get_temp_file()

        with open_(input_map_file) as fid, \
             shelve.open(input_map) as d_smi_inputs:
            reader = csv.reader(fid)
            
            for smi, *ligands in reader:
                d_smi_inputs[smi] = ligands

        return input_map
        
def distribute_and_flatten(
        xs_yss: List[Tuple[T, List[U]]]) -> List[Tuple[T, U]]:
    """Distribute x over a list ys for each item in a list of 2-tuples and
    flatten the resulting lists.

    For each item in a list of Tuple[T, List[U]], distribute T over the List[U]
    to generate a List[Tuple[T, U]] and flatten the resulting list of lists.

    Example
    -------
    >>> xs_yys = [(1, ['a','b','c']), (2, ['d']), (3, ['e', 'f'])]
    >>> distribute_and_flatten(xs_yss)
    [(1, 'a'), (1, 'b'), (1, 'c'), (2, 'd'), (3, 'e'), (3, 'f')]

    Returns
    -------
    List[Tuple[T, U]]
        the distributed and flattened list
    """
    return list(chain(*[product([x], ys) for x, ys in xs_yss]))

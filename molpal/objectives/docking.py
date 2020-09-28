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
    docker : str
        the docking program to use
    receptor : str
        the filepath of the receptor in which to dock a molecule
    center : Tuple[float, float, float]
        the coordinates of center of the docking box
    size : Tuple[int, int, int]
        the x-, y-, and z-radii of the docking box
    njobs : int
        the number of docking jobs to run in parallel
    ncpu : int
        the number of cores to run each docking job over
    input_map : Optional[str]
        the filepath of a Shelf containing the mapping from SMILES string to
        input filepath(s), if an input map was provided. Otherwise, None
    verbose : int (Default = 0)
        the level of output to print

    Parameters
    ----------
    docker : str
    receptor : str
    center : Tuple[float, float, float]
    size : Tuple[int, int, int]
    njobs : int (Default = os.cpu_count())
    ncpu : int (Default = 1)
    input_map_file : Optional[str] (Default = None)
        a CSV file containing the mappings of SMILES strings to pre-prepared
        ligand files. The first column is the SMILES string and each column
        thereafter is the filepath of an input file corresponding to that
        SMILES string
    verbose : int (Default = 0)
    input_map_file : Optional[str] (Default = None)
    **kwargs
        additional and unused keyword arguments
    """
    def __init__(self, docker: str, receptor: str,
                 center: Tuple[float, float, float],
                 size: Tuple[int, int, int],
                 njobs: int = os.cpu_count(), ncpu: int = 1,
                 boltzmann: bool = False,
                 input_map_file: Optional[str] = None,
                 verbose: int = 0, **kwargs):
        self.docker = docker
        self.receptors = docking.preparation.prepare_receptors([receptor])
        self.center = center
        self.size = size
        self.njobs = njobs
        self.ncpu = ncpu
        self.boltzmann = boltzmann
        self.input_map_file = input_map_file
        self.verbose = verbose

        if input_map_file:
            self.input_map = self._build_input_map(input_map_file)
        else:
            self.input_map = None

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
        with shelve.open(self.input_map) as d_smi_ligands:
            ligandss = [(smi, d_smi_inputs[smi])
                         for smi in smis if smi in d_smi_ligands]
            ligands = distribute_and_flatten(ligandss)
            extra_smis = [smi for smi in smis if smi not in d_smi_ligands]

        if extra_smis:
            ligands.extend(docking.prepare_ligand(extra_smis, path=in_path))

        scores, _ = docking.docking.dock_ligands(
            ligands=ligands, docker=self.docker, receptors=self.receptors,
            center=self.center, size=self.size, ncpu=self.ncpu, path=out_path,
            boltzmann=self.boltzmann, nworkers=self.njobs, verbose=self.verbose
        )

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
                d_smi_ligands[smi] = d_smi_ligands

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

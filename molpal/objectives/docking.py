import csv
from functools import partial
import gzip
from itertools import chain, product
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar

from .base import Objective
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
    njobs : int (Default = MAX_CPU)
        the number of docking jobs to run in parallel
    ncpu : int (Default = 1)
        the number of cores to run each docking job over
    navg : Optional[int] (Default = None)
        the number of top scores to average. If None, take only the top score
    input_map_file : Optional[str] (Default = None)
        a CSV file containing the mappings of SMILES strings to pre-prepared
        ligand files. The first column is the SMILES string and each column
        thereafter is the filepath of an input file corresponding to that
        SMILES string
    d_smi_pdbqts : Optional[Dict[str, List[str]]]
        the cached input map, if feasible to hold in memory
    verbose : int (Default = 0)
        the level of output to print
    **kwargs
        additional and unused keyword arguments
    """
    try:
        MAX_CPU = len(os.sched_getaffinity(0))
    except AttributeError:
        MAX_CPU = mp.cpu_count()

    def __init__(self, docker: str, receptor: str,
                 center: Tuple[float, float, float],
                 size: Tuple[int, int, int],
                 njobs: int = MAX_CPU, ncpu: int = 1,
                 boltzmann: bool = False,
                 input_map_file: Optional[str] = None,
                 verbose: int = 0, **kwargs):
        if None in [docker, receptor, center, size]:
            raise ValueError('A DockingObjective argument was None!')

        self.docker = docker
        # handle being passed a pre-converted receptor file
        # handle receptor formats for different docking programs
        self.receptors = docking.preparation.prepare_receptors([receptor])
        self.center = center
        self.size = size
        self.njobs = njobs
        self.ncpu = ncpu
        self.boltzmann = boltzmann
        self.input_map_file = input_map_file
        self.verbose = verbose

        if input_map_file:
            if Path(input_map_file).suffix == '.gz':
                self.open_ = partial(gzip.open, mode='rt')
            else:
                self.open_ = open

            self.d_smi_pdbqts = self._cache_input_map()
        else:
            self.d_smi_pdbqts = None

        super().__init__(minimize=True)

    # @singledispatchmethod
    # def calc(self, *args, **kwargs):
    #     raise NotImplementedError

    # @calc.register
    def calc_single(self, smi_or_file: str, in_path: Optional[str] = None,
                    out_path: Optional[str] = None,
                    **kwargs) -> Dict[str, Optional[float]]:
        """Caclulate the docking score(s)

        Parameters
        ----------
        smi_or_file : str
            a single SMILES string, the filepath of a ligand supply file,
            or a file containing a single ligand
        in_path : Optional[str] (Default = None)
            the path under which input files should be written
        out_path : Optional[str] (Default = None)
            the path under which output files should be written
        *args, **kwargs
            additional and unused positional and keyword arguments

        Returns
        -------
        scores : Dict[str, Optional[float]]
            a map from SMILES string to docking score. Ligands that failed to
            dock will be scored as None
        """
        if smi_or_file is None or smi_or_file == '':
            raise ValueError(f'{smi_or_file} was None or empty!')

        ligands = docking.prepare_ligand(smi_or_file, path=in_path)
        scores, _ = docking.docking.dock_ligands(
            ligands=ligands, docker=self.docker, receptors=self.receptors,
            center=self.center, size=self.size,
            nworkers=self.njobs, ncpu=self.ncpu, boltzmann=self.boltzmann,
            path=out_path, verbose=self.verbose)
        return {
            smi: self.c * score if score else None
            for smi, score in scores.items()
        }

    # @calc.register
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
        if smis is None:
            raise ValueError('smis was None!')

        # these assume we'll find all of the ligands in the input map
        if self.d_smi_pdbqts:
            ligandss = [(smi, self.d_smi_pdbqts[smi]) for smi in smis
                          if smi in self.d_smi_pdbqts]
            ligands = distribute_and_flatten(ligandss)
        elif self.input_map_file:
            ligandss = self._search_input_map(smis)
            ligands = distribute_and_flatten(ligandss)
        # handle possibly some ligands being found but others need to be still
        # be converted
        else:
            ligands = docking.prepare_ligand(smis, path=in_path)

        scores, _ = docking.docking.dock_ligands(
            ligands=ligands, docker=self.docker, receptors=self.receptors,
            center=self.center, size=self.size, ncpu=self.ncpu, path=out_path,
            boltzmann=self.boltzmann, nworkers=self.njobs, verbose=self.verbose
        )

        return {
            smi: self.c * score if score else None
            for smi, score in scores.items()
        }

    def _search_input_map(self, smis: List[str]) -> List[Tuple[str, List[str]]]:
        smis = set(smis)
        with self.open_(self.input_map_file) as fid:
            smis_pdbqtss = []
            reader = csv.reader(fid)

            for smi, *pdbqts in reader:
                if smi in smis:
                    smis_pdbqtss.append((smi, pdbqts))

        return smis_pdbqtss

    def _cache_input_map(self) -> List[Tuple[str, List[str]]]:
        with self.open_(self.input_map_file) as fid:
            size = sum(1 for _ in fid)
            if size > 1000000:
                return None

            fid.seek(0)
            reader = csv.reader(fid)

            d_smi_pdbqts = {}
            for smi, *pdbqts in reader:
                d_smi_pdbqts[smi] = pdbqts

        return d_smi_pdbqts

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

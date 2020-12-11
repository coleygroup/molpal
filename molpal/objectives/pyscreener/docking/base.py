from abc import ABC, abstractmethod
from concurrent.futures import Executor
import csv
from functools import partial
from itertools import chain
from math import ceil, exp, log10
import os
from pathlib import Path
import timeit
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

from rdkit import Chem
from tqdm import tqdm

from molpal.objectives.pyscreener.preprocessing import pdbfix

class Screener(ABC):
    """A Screener conducts virtual screens against an ensemble of receptors.

    Classes that implement the Screener interface are responsible for
    defining the following methods:
        prepare_receptor
        prepare_from_smi
        prepare_from_file
        run_docking
        parse_ligand_results

    This is an abstract base class and cannot be instantiated.

    Attributes
    ----------
    repeats : int
        the number of times each docking run will be repeated
    score_mode : str
        the mode used to calculate a score for an individual docking run given
        multiple output scored conformations
    receptor_score_mode : str
        the mode used to calculate an overall score for a single receptor
        given repeated docking runs against that receptor
    ensemble_score_mode : str
        the mode used to calculate an overall score for an ensemble of receptors
        given multiple receptors in an ensemble
    distributed : bool
        True if the computation will parallelized over a distributed setup.
        False if the computation will parallelized over a local setup
    num_workers : int
        the number of worker processes to initialize when
        distributing computation
    ncpu : int
        the number of cores allocated to each worker process
    path : os.PathLike
        the path under which input and output folders will be placed
    in_path : os.PathLike
        the path under which all prepared input files will be placed
    out_path : os.PathLike
        the path under which all generated output will be placed
    verbose : int
        the level of output this Screener should output

    Parameters
    ----------
    repeats : int (Default = 1)
    score_mode : str (Default = 'best')
    receptor_score_mode : str (Default = 'best')
    ensemble_score_mode : str (Default = 'best')
    distributed : bool (Default = False)
    num_workers : int (Default = -1)
    ncpu : int (Default = 1)
    path : Union[str, os.PathLike] (Default = '.')
    verbose : int (Default = 0)
    **kwargs
        additional and unused keyword arguments
    """
    def __init__(self, receptors: Optional[Sequence[str]] = None,
                 pdbids: Optional[Sequence[str]] = None,
                 repeats: int = 1, score_mode: str = 'best',
                 receptor_score_mode: str = 'best', 
                 ensemble_score_mode: str = 'best',
                 distributed: bool = False,
                 num_workers: int = -1, ncpu: int = 1,
                 path: str = '.', verbose: int = 0, **kwargs):
        self.path = Path(path)

        receptors = receptors or []
        if pdbids:
            receptors.extend((
                pdbfix.pdbfix(pdbid=pdbid, path=self.in_path)
                for pdbid in pdbids
            ))
        if len(receptors) == 0:
            raise ValueError('No receptors or PDBids provided!')

        self.receptors = receptors
        self.repeats = repeats
        self.score_mode = score_mode
        self.receptor_score_mode = receptor_score_mode
        self.ensemble_score_mode = ensemble_score_mode
        
        self.distributed = distributed
        self.num_workers = num_workers
        self.ncpu = ncpu

        self.verbose = verbose

        self.num_docked_ligands = 0
        
    def __len__(self) -> int:
        """The number of ligands this screener has simulated"""
        return self.num_docked_ligands

    def __call__(self, *args, **kwargs) -> Dict[str, Optional[float]]:
        return self.dock(*args, **kwargs)
    
    @property
    def path(self) -> Tuple[os.PathLike, os.PathLike]:
        """return the Screener's in_path and out_path"""
        return self.__in_path, self.__out_path
        
    @path.setter
    def path(self, path: str):
        """set both the in_path and out_path under the input path"""
        path = Path(path)
        self.in_path = path / 'inputs'
        self.out_path = path / 'outputs'

    @property
    def in_path(self) -> os.PathLike:
        return self.__in_path

    @in_path.setter
    def in_path(self, path: str):
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True)
        self.__in_path = path

    @property
    def out_path(self) -> os.PathLike:
        return self.__out_path

    @out_path.setter
    def out_path(self, path: str):
        path = Path(path)
        if not path.is_dir():
            path.mkdir(parents=True)
        self.__out_path = path

    def dock(self, *smis_or_files: Iterable,
             full_results: bool = False,
             **kwargs) -> Dict[str, Optional[float]]:
        """dock the ligands contained in sources

        NOTE: the star operator, *, in the function signature.
              If intending to pass multiple filepaths as an iterable, first 
              unpack the iterable in the function call by prepending a *.
              If passing multiple SMILES strings, either option is acceptable,
              but it is much more efficient to NOT unpack the iterable.

        Parameters
        ----------
        smis_or_files: Iterable
            an iterable of ligand sources, where each ligand source may be
            one of the following:
            - a ligand supply file,
            - a list of SMILES strings
            - a single SMILES string
        **kwargs
            keyword arguments to pass to the appropriate prepare_from_*
            function(s)

        Returns
        -------
        d_smi_score : Dict[str, Optional[float]]
            a dictionary mapping SMILES string to the best score among the
            corresponding ligands. (None if all corresponding ligands
            failed failed to dock)
        records : List[Dict]
            a list of dictionaries containing the record of every single
            docking run performed. Each dictionary contains the following keys:
            - smiles: the ligand's SMILES string
            - name: the name of the ligand
            - in: the filename of the input ligand file
            - out: the filename of the output docked ligand file
            - log: the filename of the output log file
            - score: the ligand's docking score
        """
        recordsss = self.dock_ensemble(*smis_or_files, **kwargs)

        smis_scores = []
        for ligand_results in recordsss:
            smi = ligand_results[0][0]['smiles']
            score = self.calc_ligand_score(
                ligand_results, self.receptor_score_mode,
                self.ensemble_score_mode
            )
            smis_scores.append((smi, score))

        d_smi_score = {}
        for smi_score in smis_scores:
            smi, score = smi_score
            if smi not in d_smi_score:
                d_smi_score[smi] = score
            elif score is None:
                continue
            else:
                curr_score = d_smi_score[smi]
                if curr_score is None:
                    d_smi_score[smi] = score
                else:
                    d_smi_score[smi] = min(d_smi_score[smi], score)

        if full_results:
            return d_smi_score, list(chain(*list(chain(*recordsss))))

        return d_smi_score

    def dock_ensemble(self, *smis_or_files: Iterable,
                      **kwargs) -> List[List[List[Dict]]]:
        """Run the docking program with the ligands contained in *smis_or_files

        NOTE: the zip operator, *, in the function signature. If intending to
              pass multiple filepaths as an iterable, first unpack the iterable
              in the function call by prepending a *

        Parameters
        ----------
        smis_or_files: Iterable
            an iterable of ligand sources, where each ligand source may be
            one of the following:
            - a ligand supply file
            - a list of SMILES strings
            - a single SMILES string
        **kwargs
            keyword arguments to pass to the appropriate prepare_from_*
            function(s)

        Returns
        -------
        recordsss : List[List[List[Dict]]]
            an NxMxO list of dictionaries where each dictionary is a record of an individual docking run and:
            - N is the number of total ligands that will be docked
            - M is the number of receptors each ligand is docked against
            - O is the number of times each docking run is repeated.
            Each dictionary contains the following keys:
            - smiles: the ligand's SMILES string
            - name: the name of the ligand
            - in: the filename of the input ligand file
            - out: the filename of the output docked ligand file
            - log: the filename of the output log file
            - score: the ligand's docking score

        """
        begin = timeit.default_timer()

        ligands = self.prepare_ligands(*smis_or_files, **kwargs)
        recordsss = self.run_docking(ligands)
        # recordsss = self.parse_docking(recordsss_unparsed)

        self.num_docked_ligands += len(recordsss)

        total = timeit.default_timer() - begin

        mins, secs = divmod(int(total), 60)
        hrs, mins = divmod(mins, 60)
        if self.verbose > 0 and len(recordsss) > 0:
            print(f'  Time to dock {len(recordsss)} ligands:',
                f'{hrs:d}h {mins:d}m {secs:d}s ' +
                f'({total/len(recordsss):0.3f} s/ligand)', flush=True)

        return recordsss

    @abstractmethod
    def run_docking(self, ligands: Sequence[Tuple[str, str]]
                   ) -> List[List[List[Dict]]]:
        """Run the docking simulations for the input ligands
        
        Parameter
        ----------
        ligands : Sequence[Tuple[str, str]]
            a sequence of tuples containing a ligand's SMILES string and the 
            filepath of the corresponding input file
        
        Returns
        -------
        List[List[List[Dict]]]
            an NxMxO list of dictionaries where each individual dictionary is a 
            record of an individual docking run and
            N is the number of ligands contained in the ligand sources
            M is the number of receptors in the ensemble against which each 
                ligand should be docked
            O is the number of times each docking run should be repeated
            NOTE: the records contain a 'score' that is None for each entry
                  as the log/out files must first be parsed to obtain the value
        """

    def parse_docking(self, ligs_recs_reps: List[List[List[Dict]]]
                     ) -> List[List[List[Dict]]]:
        """Parse the results of all the docking simulations and update the
        records accordingly
        
        Parameter
        ----------
        ligs_recs_reps : List[List[List[Dict]]]
            an NxMxO list of dictionaries where each individual dictionary is a 
            record of an individual docking run and
            N is the number of ligands contained in the ligand sources
            M is the number of receptors in the ensemble against which each 
                ligand should be docked
            O is the number of times each docking run should be repeated
        
        Returns
        -------
        ligs_recs_reps : List[List[List[Dict]]]
            the same List as the input argument, but with the 'score' key of
            record updated to reflect the desired score parsed
            from each docking run
        """
        parse_ligand_results_ = partial(self.parse_ligand_results,
                                        score_mode=self.score_mode)
        CHUNKSIZE = 128
        with self.Pool(self.distributed,
                       self.num_workers, self.ncpu, True) as client:
            ligs_recs_reps = list(tqdm(
                client.map(parse_ligand_results_, ligs_recs_reps, 
                           chunksize=CHUNKSIZE), total = len(ligs_recs_reps),
                desc='Parsing results', unit='ligand', smoothing=0.
            ))
        
        return ligs_recs_reps

    @staticmethod
    @abstractmethod
    def parse_ligand_results(recs_reps: List[List[Dict]],
                             score_mode: str = 'best') -> List[List[Dict]]:
        """Parse the results of the docking simulations for a single ligand
        
        Parameter
        ----------
        recs_reps : List[List[Dict]]
            an MxO list of list of dictionaries where each individual 
            dictionary is a record of an individual docking run and
            M is the number of receptors in the ensemble against which each 
                ligand should be docked
            O is the number of times each docking run should be repeated
        
        Returns
        -------
        recs_reps : List[List[Dict]]
            the same List as the input argument, but with the 'score' key of
            record updated to reflect the desired score parsed
            from each docking run
        """

    @property
    def receptors(self):
        return self.__receptors

    @receptors.setter
    def receptors(self, receptors):
        receptors = [self.prepare_receptor(receptor) for receptor in receptors]
        receptors = [receptor for receptor in receptors if receptor is not None]
        if len(receptors) == 0:
            raise RuntimeError('Preparation failed for all receptors!')
        self.__receptors = receptors
    
    @abstractmethod
    def prepare_receptor(self, *args, **kwargs):
        """Prepare a receptor input file for the docking software"""

    @staticmethod
    @abstractmethod
    def prepare_from_smi(*args, **kwargs):
        """Prepare a ligand input file from a SMILES string"""

    @staticmethod
    @abstractmethod
    def prepare_from_file(*args, **kwargs):
        """Prepare a ligand input file from an input file"""

    def prepare_ligands(self, *smis_or_files,
                        path: Optional[str] = None, **kwargs):
        path = path or self.in_path
        return list(chain(*(
            self._prepare_ligands(source, i+len(self), path, **kwargs)
            for i, source in enumerate(smis_or_files)
        )))

    def _prepare_ligands(self, source, i: int,
                         path: Optional[str] = None, **kwargs):
        if isinstance(source, str):
            p_ligand = Path(source)

            if not p_ligand.exists():
                return [self.prepare_from_smi(source, f'ligand_{i}', path)]

            if p_ligand.suffix == '.csv':
                return self.prepare_from_csv(source, **kwargs)
            if p_ligand.suffix == '.smi':
                return self.prepare_from_supply(source, **kwargs)
            if p_ligand.suffix == '.sdf':
                if kwargs['use_3d']:
                    return self.prepare_from_file(source, path=path,
                                                  **kwargs)
                else:
                    return self.prepare_from_supply(source, **kwargs)
            
            return self.prepare_from_file(source, path=path, **kwargs)

        if isinstance(source, Sequence):
            return self.prepare_from_smis(source, **kwargs)
        
        raise TypeError('Arg "source" must be of type str or ', 
                        f'Sequence[str]. Got: {type(source)}')

    def prepare_from_smis(self, smis: Sequence[str],
                          names: Optional[Sequence[str]] = None, 
                          start: int = 0, nconvert: Optional[int] = None,
                          **kwargs) -> List[Tuple]:
        """Convert the list of SMILES strings to their corresponding input files

        Parameters
        ----------
        smis : Sequence[str]
            a sequence of SMILES strings
        names : Optional[Sequence[str]] (Default = None)
            a parallel sequence of names for each ligand
        start : int (Default = 0)
            the index at which to start ligand preparation
        nconvert : Optional[int] (Default = None)
            the number of ligands to convert. If None, convert all ligands
        **kwargs
            additional and unused keyword arguments

        Returns
        -------
        ligands : List[Tuple]
            a list of tuples containing a ligand's SMILES string and the 
            filepath of the corresponding input file
        """
        begin = timeit.default_timer()
        
        stop = min(len(smis), start+nconvert) if nconvert else len(smis)

        if names is None:
            width = ceil(log10(len(smis))) + 1
            names = (f'ligand_{i:0{width}}' for i in range(start, stop))
        else:
            # could theoretically handle empty strings
            names = names[start:stop]
        smis = smis[start:stop]
        paths = (self.in_path for _ in range(len(smis)))

        CHUNKSIZE = 1
        with self.Pool(self.distributed, self.num_workers,
                       self.ncpu, True) as client:
            ligands = client.map(self.prepare_from_smi, smis, names, paths, 
                                 chunksize=CHUNKSIZE)
            ligands = [
                ligand for ligand in tqdm(
                    ligands, total=len(smis), desc='Preparing ligands', 
                    unit='ligand', smoothing=0.
                ) if ligand
            ]
        
        total = timeit.default_timer() - begin
        if self.verbose > 1:
            m, s = divmod(int(total), 60)
            h, m = divmod(m, 60)
            if len(ligands) > 0:
                print(f'    Time to prepare {len(ligands)} ligands: ',
                      f'{h}h {m}m {s}s ({total/len(ligands):0.4f} s/ligand)', 
                      flush=True)
            
        return ligands

    def prepare_from_csv(self, csv_filename: str, title_line: bool = True,
                         smiles_col: int = 0, name_col: Optional[int] = None,
                         start: int = 0, nconvert: Optional[int] = None,
                         **kwargs) -> List[Tuple]:
        """Prepare the input files corresponding to the SMILES strings
        contained in a CSV file

        Parameters
        ----------
        csv_filename : str
            the filename of the CSV file containing the ligands to convert
        title_line : bool (Default = True)
            does the CSV file contain a title line?
        smiles_col : int (Default = 0)
            the column containing the SMILES strings
        name_col : Optional[int] (Default = None)
            the column containing the molecule name
        start : int (Default = 0)
            the index at which to start conversion
        nconvert : Optional[int] (Default = None)
            the number of ligands to convert. If None, convert all molecules
        **kwargs
            additional and unused keyword arguments

        Returns
        -------
        ligands : List[Tuple]
            a list of tuples containing a ligand's SMILES string and the 
            filepath of the corresponding input file. Files are named 
            <compound_id>.<suffix> if compound_id property exists in the 
            original supply file. Otherwise, they are named:
                lig0.<suffix>, lig1.<suffix>, ...
        """
        with open(csv_filename) as fid:
            reader = csv.reader(fid)
            if title_line:
                next(reader)

            if name_col is None:
                smis = [row[smiles_col] for row in reader]
                names = None
            else:
                smis_names = [(row[smiles_col], row[name_col])
                              for row in reader]
                smis, names = zip(*smis_names)
        
        return self.prepare_from_smis(smis, names=names,
                                      start=start, nconvert=nconvert)

    def prepare_from_supply(self, supply: str,
                            id_prop_name: Optional[str] = None,
                            start: int = 0, nconvert: Optional[int] = None,  
                            **kwargs) -> List[Tuple]:
        """Prepare the input files corresponding to the molecules contained in 
        a molecular supply file

        Parameters
        ----------
        supply : str
            the filename of the SDF or SMI file containing
            the ligands to convert
        id_prop_name : Optional[str]
            the name of the property containing the ID, if one exists
            (e.g., "CatalogID", "Chemspace_ID", "Name", etc...)
        start : int (Default = 0)
            the index at which to start ligand conversion
        nconvert : Optional[int] (Default = None)
            the number of ligands to convert. If None, convert all molecules
        **kwargs
            additional and unused keyword arguments

        Returns
        -------
        ligands : List[Tuple[str, str]]
            a list of tuples containing a ligand's SMILES string and the 
            filepath of the corresponding input file. Files are named 
            <compound_id>.<suffix> if compound_id property exists in the 
            original supply file. Otherwise, they are named:
                lig0.<suffix>, lig1.<suffix>, ...
        """
        p_supply = Path(supply)
        if p_supply.suffix == '.sdf':
            mols = Chem.SDMolSupplier(supply)
        elif p_supply.suffix == '.smi':
            mols = Chem.SmilesMolSupplier(supply)
        else:
            raise ValueError(
                f'input file: "{supply}" does not have .sdf or .smi extension')

        smis = []
        names = None

        if id_prop_name:
            names = []
            for mol in mols:
                if mol is None:
                    continue

                smis.append(Chem.MolToSmiles(mol))
                names.append(mol.GetProp(id_prop_name))
        else:
            for mol in mols:
                if mol is None:
                    continue

                smis.append(Chem.MolToSmiles(mol))

        return self.prepare_from_smis(smis, names=names,
                                      start=start, nconvert=nconvert)

    @staticmethod
    def calc_ligand_score(ligand_results: List[List[Dict]],
                          receptor_score_mode: str = 'best',
                          ensemble_score_mode: str = 'best') -> Optional[float]:
        """Calculate the overall score of a ligand given all of its docking
        runs

        Parameters
        ----------
        ligand_results : List[List[Dict]]
            an MxO list of list of dictionaries where each individual dictionary is a record of an individual docking run and
            M is the number of receptors the ligand was docked against
            O is the number of times each docking run was repeated
        receptor_score_mode : str (Default = 'best')
            the mode used to calculate the overall score for a given receptor
            pose with multiple, repeated runs
        ensemble_score_mode : str (Default = 'best')
            the mode used to calculate the overall score for a given ensemble
            of receptors

        Returns
        -------
        ensemble_score : Optional[float]
            the overall score of a ligand's ensemble docking. None if no such
            score was calculable
        
        See also
        --------
        calc_score
            for documentation on possible values for receptor_score_mode
            and ensemble_score_mode
        """
        receptor_scores = []
        for receptor in ligand_results:
            successful_rep_scores = [
                repeat['score']
                for repeat in receptor if repeat['score'] is not None
            ]
            if successful_rep_scores:
                receptor_scores.append(Screener.calc_score(
                    successful_rep_scores, receptor_score_mode
                ))

        receptor_scores = [score for score in receptor_scores]
        if receptor_scores:
            ensemble_score = Screener.calc_score(
                receptor_scores, ensemble_score_mode)
        else:
            ensemble_score = None
        
        return ensemble_score
    
    @staticmethod
    def calc_score(scores: Sequence[float], score_mode: str = 'best') -> float:
        """Calculate an overall score from a sequence of scores

        Parameters
        ----------
        scores : Sequence[float]
        score_mode : str (Default = 'best')
            the method used to calculate the overall score
            Choices:
                'best' - return the top score
                'avg' - return the average of the scores
                'boltzmann' - return the boltzmann average of the scores

        Returns
        -------
        score : float
        """
        scores = sorted(scores)
        if score_mode in ('best', 'top'):
            score = scores[0]
        elif score_mode in ('avg', 'mean'):
            score = sum(score for score in scores) / len(scores)
        elif score_mode == 'boltzmann':
            Z = sum(exp(-score) for score in scores)
            score = sum(score * exp(-score) / Z for score in scores)
        
        return score
    
    @staticmethod
    def Pool(distributed: bool = False, num_workers: int = -1, ncpu: int = 1,
             all_cores: bool = False) -> Type[Executor]:
        """build a process pool to parallelize computation over

        Parameters
        ----------
        distributed : bool (Default = False)
            whether to return a distributed or a local process pool
        num_workers : int (Default = -1)
            if distributed is True, then this argument is ignored. If False,
            then it should be equal to the total number of worker processes
            desired. Using a value of -1 will spawn as many worker processes
            as cores available on this machine.
            NOTE: this is usually not a good idea and it's much better to
                  specify the number of processes explicitly.
        ncpu : int (Default = 1)
            if distributed is True, then this argument should be the number of 
            cores allocated to each worker. if False, then this should be the
            number of cores that is desired to be allocated to each worker.
            NOTE: this is an implicit argument because Screener.dock() will   
                  make subprocess calls to progams that themselves can utilize 
                  multiple cores. It will not actually assign <ncpu> cores to 
                  each worker process.
        all_cores : bool (Default = False)
            whether to initialize as many processes as cores available
            (= num_workers * ncpu).
        
        Returns
        -------
        Executor
            the initialized process pool
        
        Notes
        -----
        in some cases, as shown in the examples below, the values specified for
        num_workers and ncpu will be inconsequential. Regardless, it is good
        practice for this function to always be called the same way, with only
        all_cores changing, depending on the context in which the initialized Executor will be used

        Ex. 1
        -----
        Given: a single machine with 16 cores, screening using vina-type
               docking software (via the docking.Vina class)
        the function should be called with distributed=False, all_cores=False, 
        and both num_workers and ncpu should be specified such that the product 
        of the two is equal to 16.
        Choices: (1, 16), (2, 8), (4, 4), (8, 2), and (16, 1). You will often have to determine the optimal values empirically.

        Ex. 2
        -----
        Given: a cluster of machines where you've requested resources for 8
               tasks with 2 cores each. The software was then initialized with
               8 separate MPI processes and screening using vina-type docking
               software is to be performed.
        the function should be called with distributed=True and all_cores=False
        (neither num_workers or ncpu needs to be specified)

        Ex. 3
        -----
        Given: a single machine with 16 cores, and pure python code is to be
               executed in parallel
        the function should be called with distributed=False, all_cores=True,
        and both num_workers and ncpu should be specified such that the product 
        of the two is equal to 16.
        Choices: see Ex. 1
        """
        if distributed:
            from mpi4py import MPI
            from mpi4py.futures import MPIPoolExecutor as Pool

            num_workers = MPI.COMM_WORLD.size
        else:
            from concurrent.futures import ProcessPoolExecutor as Pool
            if num_workers == -1:
                try:
                    num_workers = len(os.sched_getaffinity(0))
                except AttributeError:
                    num_workers = os.cpu_count()

        if all_cores:
            num_workers *= ncpu

        return Pool(max_workers=num_workers)

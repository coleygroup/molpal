"""This module contains the Explorer class, which is an abstraction
for batched, Bayesian optimization."""

from collections import deque
import csv
import heapq
from itertools import zip_longest
import os
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from . import acquirer
from . import encoders
from . import models
from . import objectives
from . import pools

class Explorer:
    """An Explorer explores a pool of inputs using Bayesian optimization

    Attributes (constant)
    ----------
    name : str
        the name this explorer will use for all outputs
    pool : MoleculePool
        the pool of inputs to explore
    acq : Acquirer
        an acquirer which selects molecules to explore next using a prior
        distribution over the inputs
    obj : Objective
        an objective calculates the objective function of a set of inputs
    retrain_from_scratch : bool
        whether the model will be retrained from scratch at each iteration.
        If False, train the model online. 
        NOTE: The definition of 'online' is model-specific.
    recent_avgs : Deque[float]
        a queue containing the <window_size> most recent averages
    delta : float
        the minimum acceptable fractional difference between the current 
        average and the moving average in order to continue exploration
    max_epochs : int
        the maximum number of batches to explore
    root : str
        the directory under which to organize all outputs
    tmp : str
        the temporary directory of the filesystem
    write_final : bool
        whether the list of explored inputs and their scores should be written
        to a file at the end of exploration
    write_intermediate : bool
        whether the list of explored inputs and their scores should be written
        to a file after each round of exploration
    write_preds : bool
        whether the predictions should be written after each exploration batch
    verbose : int
        the level of output this Explorer prints

    Attributes (stateful)
    ----------
    model : Model
        a model that generates a posterior distribution over the inputs using
        observed data
    epoch : int
        the current epoch of exploration
    scores : Dict[str, float]
        a dictionary mapping a molecule's SMILES string to its corresponding
        objective function value
    failed : Dict[str, None]
        a dictionary containing the inputs for which the objective function
        failed to evaluate
    new_scores : Dict[str, float]
        a dictionary mapping a molecule's SMILES string to its corresponding
        objective function value for the most recent batch of labeled inputs
    new_model : bool
        whether the predictions are currently out-of-date with the model
    top_k_avg : float
        the average of the top-k explored inputs
    y_preds : List[float]
        a parallel list to the pool containing the mean predicted score
        for an input
    y_vars : List[float]
        a parallel list to the pool containing the variance in the predicted
        score for an input. Will be empty if model does not provide variance
    

    Properties
    ----------
    k : int
    max_explore : int

    Parameters
    ----------
    name : str
    k : Union[int, float] (Default = 0.01)
    window_size : int (Default = 3)
        the number of top-k averages from which to calculate a moving average
    delta : float (Default = 0.01)
    max_epochs : int (Default = 50)
    max_explore : Union[int, float] (Default = 1.)
    root : str (Default = '.')
    tmp : str (Default = $TMP or '.')
    write_final : bool (Default = True)
    write_intermediate : bool (Default = False)
    save_preds : bool (Default = False)
    retrain_from_scratch : bool (Default = False)
    previous_scores : Optional[str] (Default = None)
        the filepath of a CSV file containing previous scoring data which will
        be treated as the initialization batch (instead of randomly selecting
        from the bool.)
    scores_csvs : Optional[List[str]] (Default = None)
        a list of filepaths containing CSVs with previous scoring data. These
        CSVs will be read in and the model trained on the data in the order
        in which the CSVs are provide. This is useful for mimicking the
        intermediate state of a previous Explorer instance
    verbose : int (Default = 0)
    **kwargs
        the keyword arguments to initialize an Encoder, MoleculePool, Acquirer, 
        Model, and Objective class

    Raises
    ------
    ValueError
        if k is less than 0
        if max_explore is less than 0
    """
    def __init__(self, name: str,
                 k: Union[int, float] = 0.01,
                 window_size: int = 3, delta: float = 0.01,
                 max_epochs: int = 50, max_explore: Union[int, float] = 1.,
                 root: str = '.', tmp: str = os.environ.get('TMP', '.'),
                 write_final: bool = True, write_intermediate: bool = False,
                 save_preds: bool = False, 
                 retrain_from_scratch: bool = False,
                 previous_scores: Optional[str] = None,
                 scores_csvs: Optional[List[str]] = None,
                 verbose: int = 0, **kwargs):
        self.name = name; kwargs['name'] = name
        self.verbose = verbose; kwargs['verbose'] = verbose
        self.root = root
        self.tmp = tmp

        self.enc = encoders.encoder(**kwargs)
        self.pool = pools.pool(enc=self.enc, path=self.tmp, **kwargs)
        self.acq = acquirer.Acquirer(size=len(self.pool), **kwargs)

        if self.acq.metric_type == 'thompson':
            kwargs['dropout_size'] = 1
        self.model = models.model(input_size=len(self.enc), **kwargs)
        self.obj = objectives.objective(**kwargs)

        self._validate_acquirer()

        self.acq.stochastic_preds = 'stochastic' in self.model.provides
        self.retrain_from_scratch = retrain_from_scratch

        # if k < 0:
        #     raise ValueError(f'k(={k}) must be greater than 0!')
        # if isinstance(k, float):
        #     k = int(k * len(self.pool))
        # self.k = min(k, len(self.pool))
        self.k = k
        self.delta = delta

        # if max_explore < 0.:
        #     raise ValueError(f'max_explore must be greater than 0!')
        # if isinstance(max_explore, float):
        #     max_explore = int(len(self.pool) * max_explore)
        # self.max_explore = min(max_explore, len(self.pool))
        self.max_explore = max_explore
        self.max_epochs = max_epochs

        self.write_final = write_final
        self.write_intermediate = write_intermediate
        self.save_preds = save_preds

        # state variables (not including model)
        self.epoch = 0
        self.scores = {}
        self.failed = {}
        self.new_scores = {}
        self.new_model = None
        self.recent_avgs = deque(maxlen=window_size)
        self.top_k_avg = None
        self.y_preds = None
        self.y_vars = None

        if previous_scores:
            self.load_scores(previous_scores)
        elif scores_csvs:
            self.load(scores_csvs)

    @property
    def k(self) -> int:
        """The number of top-scoring inputs from which to determine
        the average.
        
        Returned as an integer but can be set either as an integer or as
        a fraction of the pool.
        NOTE: Specifying either a fraction greater than 1 or or a number larger
              than the pool size will default to using the full pool.
        l"""
        k = self.__k
        if isinstance(k, float):
            k = int(k * len(self.pool))
            
        return min(k, len(self.pool))

    @k.setter
    def k(self, k: Union[int, float]):
        if k < 0:
            raise ValueError(f'k(={k}) must be greater than 0!')
        self.__k = k

    @property
    def max_explore(self) -> int:
        """The maximum number of inputs to explore
        
        Returned as an integer but can be set either as an integer or as
        a fraction of the pool.
        NOTE: Specifying either a fraction greater than 1 or or a number larger
              than the pool size will default to using the full pool.
        """
        max_explore = self.__max_explore
        if isinstance(max_explore, float):
            max_explore = int(max_explore * len(self.pool))
        
        return max_explore
    
    @max_explore.setter
    def max_explore(self, max_explore: Union[int, float]):
        if max_explore < 0.:
            raise ValueError(f'max_explore(={max_explore}) must be greater than 0!')

        self.__max_explore = max_explore

    def explore(self):
        self.run()

    def run(self):
        """Explore the MoleculePool until the stopping condition is met"""
        
        if not self.epoch:
            print('Starting Exploration ...')
            avg = self.explore_initial()
        else:
            print(f'Resuming Exploration at epoch {self.epoch}...')
            avg = self.explore_batch()

        while not self.stopping_condition():
            if self.verbose > 0:
                k = min(len(self.scores), self.k)
                print(f'Current average of top {k}: {avg:0.3f}',
                      'Continuing exploration ...', flush=True)

            avg = self.explore_batch()

        print('Finished exploring!')
        print(f'Explored a total of {len(self)} molecules',
              f'over {self.epoch} iterations')
        print(f'Final average of top {self.k}: {avg:0.3f}')
        print(f'Final averages')
        print(f'--------------')
        for k in [0.0001, 0.0005, 0.001, 0.005]:
            print(f'top {k*100:0.2f}%: {self.avg(k):0.3f}')
        
        if self.write_final:
            self.write_scores(final=True)

    def __len__(self) -> int:
        """The number of inputs that have been explored"""
        return len(self.scores) + len(self.failed)

    def explore_initial(self) -> float:
        """Perform an initial round of exploration

        Must be called before explore_batch()

        Returns
        -------
        avg : float
            the current average score
        """
        ligands = self.acq.acquire_initial(
            xs=self.pool.smis(),
            cluster_ids=self.pool.cluster_ids(),
            cluster_sizes=self.pool.cluster_sizes,
        )

        new_scores = self.obj.calc(
            ligands, 
            in_path=f'{self.tmp}/{self.name}/input/initial',
            out_path=f'{self.tmp}/{self.name}/output/initial'
        )
        self._clean_and_update_scores(new_scores)

        self.top_k_avg = self.avg()
        if len(self.scores) >= self.k:
            self.recent_avgs.append(self.top_k_avg)

        if self.write_intermediate:
            self.write_scores()
        
        self.epoch += 1            

        return self.top_k_avg

    def explore_batch(self) -> float:
        """Perform a round of exploration

        Returns
        -------
        avg : float
            the current average score

        Raises
        ------
        InvalidExplorationError
            if before explore_initial or load_scores
        """
        if self.epoch == 0:
            raise InvalidExplorationError(
                'Cannot call explore_batch without initialization!')

        if len(self.scores) >= len(self.pool):
            # this needs to be reconsidered for transfer learning type approach
            self.epoch += 1
            return self.top_k_avg

        self._update_model()
        self._update_predictions()

        ligands = self.acq.acquire_batch(
            xs=self.pool.smis(), y_means=self.y_preds, y_vars=self.y_vars,
            explored={**self.scores, **self.failed},
            cluster_ids=self.pool.cluster_ids(),
            cluster_sizes=self.pool.cluster_sizes, epoch=self.epoch,
        )

        new_scores = self.obj.calc(
            ligands,
            in_path=f'{self.tmp}/{self.name}/input/{self.epoch}',
            out_path=f'{self.tmp}/{self.name}/output/{self.epoch}'
        )
        self._clean_and_update_scores(new_scores)

        self.top_k_avg = self.avg()
        if len(self.scores) >= self.k:
            self.recent_avgs.append(self.top_k_avg)

        if self.write_intermediate:
            self.write_scores()
        
        self.epoch += 1

        return self.top_k_avg

    def avg(self, k: Optional[float] = None) -> float:
        """Calculate the average of the top k molecules
        
        Parameters
        ----------
        k : Union[int, float, None] (Default = None)
            the number of molecules to consider when calculating the
            average, expressed either as a specific number or as a 
            fraction of the pool. If None, use self.k. If the value specified 
            is greater than the number of successfully evaluated inputs, 
            default to that value.
        """
        k = k or self.k
        if isinstance(k, float):
            k = int(k * len(self.pool))
        k = min(k, len(self.scores))

        if k == len(self.pool):
            return sum(score for score in self.scores.items()) / k
        
        return sum(score for smi, score in self.top_explored(k)) / k

    def stopping_condition(self) -> bool:
        """Is the stopping condition met?

        Returns
        -------
        bool
            whether the stopping condition has been met
            True:
                a. explored the entire pool
                b. exceeded the maximum number of iterations
                c. exceeded the maximum number of objective function evaluations
                d. the average has not improved by at least some fraction delta 
                   of the top-k moving average
            False:
                a. has not explored at least k molecules
                b. has not completed enough epochs to calculate a moving average
                c. the current top-k average is better than the top-k moving
                   average by at least some fraction delta
        """
        if self.epoch > self.max_epochs:
            return True
        if len(self.scores) >= self.max_explore:
            return True

        if len(self.recent_avgs) < self.recent_avgs.maxlen:
            return False

        moving_avg = sum(self.recent_avgs) / len(self.recent_avgs)
        return (self.top_k_avg - moving_avg) / moving_avg <= self.delta

    def top_explored(self, k: Optional[float] = None) -> List[Tuple[str,float]]:
        """Get the top-k explored molecules"""
        k = k or self.k
        if isinstance(k, float):
            k = int(k * len(self.pool))
        k = min(k, len(self.scores))

        if k / len(self.scores) < 0.8:
            return heapq.nlargest(k, self.scores.items(), key=itemgetter(1))
        
        return sorted(self.scores.items(), key=itemgetter(1), reverse=True)

    def top_preds(self, k: Optional[int] = None) -> List[Tuple[str, float]]:
        """Get the current top predicted molecules and their scores"""
        k = k or self.k

        selected = []
        for x, y in zip(self.pool.smis(), self.y_preds):
            if len(selected) < k:
                heapq.heappush(selected, (y, x))
            else:
                heapq.heappushpop(selected, (y, x))

        return [(x, y) for y, x in selected]

    def write_scores(self, m: Union[int, float] = 1., 
                     final: bool = False) -> None:
        """Write the top M scores to a CSV file

        Writes a CSV file of the top-k explored inputs with the input ID and
        the respective objective function value.

        Parameters
        ----------
        m : Union[int, float] (Default = 1.)
            the number of top inputs to write. By default, writes all inputs
        final : bool (Default = False)
            whether the explorer has finished.
        """
        if isinstance(m, float):
            m = int(m * len(self))
        m = min(m, len(self))

        p_data = Path(f'{self.root}/{self.name}/data')
        if not p_data.is_dir():
            p_data.mkdir(parents=True)

        if final:
            m = len(self)
            p_scores = Path(p_data / f'all_explored_final.csv')
        else:
            p_scores = Path(p_data / f'top_{m}_explored_iter_{self.epoch}.csv')

        top_m = self.top_explored(m)

        with open(p_scores, 'w') as fid:
            writer = csv.writer(fid)
            writer.writerow(['smiles', 'score'])
            writer.writerows(top_m)
        
        if self.verbose > 0:
            print(f'Results were written to "{p_scores}"')

    def load_scores(self, previous_scores: str) -> None:
        """Load the scores CSV located at saved_scores"""

        if self.verbose > 0:
            print(f'Loading scores from "{previous_scores}" ... ', end='')

        self.scores = self._read_scores(previous_scores)
        
        # skip initial batch acquisition (because it doesn't make sense)
        if self.epoch == 0:
            self.epoch = 1
        
        if self.verbose > 0:
            print('Done!')

    def load(self, scores_csvs: List[str]) -> None:
        """Mimic the intermediate state of a previous explorer run by loading
        the data from the list of output files"""

        if self.verbose > 0:
            print(f'Mimicking previous state ... ', end='')

        for scores_csv in scores_csvs:
            scores = self._read_scores(scores_csv)
            self.new_scores = dict(set(self.scores) ^ set(scores))
            if not self.retrain_from_scratch:
                # training at each iteration is a waste if retraining from
                # scratch (because it's not acquiring anything here)
                self._update_model()
            self.scores = scores
            self.epoch += 1

            self.top_k_avg = self.avg()
            if len(self.scores) >= self.k:
                self.recent_avgs.append(self.top_k_avg)
        
        if self.retrain_from_scratch:
            self._update_model()

        self._update_predictions()

        if self.verbose > 0:
            print('Done!')

    def write_preds(self) -> None:
        preds_path = Path(f'{self.root}/{self.name}/preds')
        if not preds_path.is_dir():
            preds_path.mkdir(parents=True)

        with open(f'{preds_path}/preds_iter_{self.epoch}.csv', 'w') as fid:
            writer = csv.writer(fid)
            writer.writerow(['smiles', 'predicted_score[, predicted_variance]'])
            writer.writerows(
                zip_longest(self.pool.smis(), self.y_preds, self.y_vars)
            )
    
    def _clean_and_update_scores(self, new_scores: Dict[str, Optional[float]]):
        """Remove the None entries from new_scores and update new_scores, 
        scores, and failed attributes accordingly

        Side effects
        ------------
        (sets) self.new_scores : Dict[str, float]
            sets self.new_scores to new_scores without the None entries
        (mutates) self.scores : Dict[str, float]
            adds the non-None entries from new_scores
        (mutates) self.failed : Dict[str, None]
            a dictionary storing the inputs for which scoring failed
        """
        for x, y in new_scores.items():
            if y is None:
                self.failed[x] = y
            else:
                self.new_scores[x] = y
                self.scores[x] = y

    def _update_model(self) -> None:
        """Update the prior distribution to generate a posterior distribution

        Side effects
        ------------
        (mutates) self.model : Type[Model]
            updates the model with new data, if necessary
        (sets) self.new_scores : Dict[str, Optional[float]]
            reinitializes self.new_scores to an empty dictionary
        (sets) self.new_model : bool
            sets self.new_model to True, indicating that the predictions are
            now out-of-date compared to the model
        """
        if len(self.new_scores) == 0:
            # only update model if there are new data
            self.new_model = False
            return

        if self.retrain_from_scratch:
            xs, ys = zip(*self.scores.items())
            retrain = True
        else:
            xs, ys = zip(*self.new_scores.items())
            retrain = False

        self.model.train(xs, ys, retrain=retrain,
                         featurize=self.enc.encode_and_uncompress)
        self.new_scores = {}
        self.new_model = True

    def _update_predictions(self) -> None:
        """Update the predictions over the pool with the new model

        Side effects
        ------------
        (sets) self.y_preds : List[float]
            a list of floats parallel to the pool inputs containing the mean
            predicted score for each input
        (sets) self.y_vars : List[float]
            a list of floats parallel to the pool inputs containing the
            predicted variance for each input
        (sets) self.new_model : bool
            sets self.new_model to False, indicating that the predictions are
            now up-to-date with the current model
        """
        if not self.new_model:
            return

        self.y_preds, self.y_vars = self.model.apply(
            x_ids=self.pool.smis(), 
            x_feats=self.pool.fps(), 
            batched_size=None, size=len(self.pool), 
            mean_only='vars' not in self.acq.needs
        )

        self.new_model = False
        
        if self.save_preds:
            self.write_preds()

    def _validate_acquirer(self) -> None:
        if self.acq.needs > self.model.provides:
            raise IncompatibilityError(
                f'{self.acq.metric_type} metric needs {self.acq.needs} but '
                + f'{self.model.type_} only provides {self.model.provides}')

    def _read_scores(self, scores_csv: str) -> Dict:
        """read the scores contained in the file located at scores_csv"""
        with open(scores_csv) as fid:
            reader = csv.reader(fid)
            next(reader)
            scores = {row[0]: float(row[1]) for row in reader}
        
        return scores

class InvalidExplorationError(Exception):
    pass

class AcquisitionError(Exception):
    pass

class IncompatibilityError(Exception):
    pass

"""This module contains the Acquirer class, which is used to gather inputs for
a subsequent round of exploration based on prior prediction data."""
import heapq
from itertools import chain
import math
from timeit import default_timer
from typing import (Dict, Iterable, List, Mapping,
                    Optional, Set, TypeVar, Union)

import numpy as np
import ray
from tqdm import tqdm
from molpal.objectives import Objective, MultiObjective
from molpal.acquirer.pareto import Pareto
from molpal.acquirer import metrics
from molpal.featurizer import feature_matrix, Featurizer
from molpal.pools.cluster import cluster_fps


T = TypeVar('T')


class Acquirer:
    """An Acquirer acquires inputs from an input pool for exploration.

    Attributes
    ----------
    size : int
        the size of the pool this acquirer will work on
    metric : str (Default = 'greedy')
        the alias of the metric to use
    epsilon : float
        the fraction of each batch that should be acquired randomly
    temp_i : Optional[float]
        the initial temperature value to use for clustered acquisition
    temp_f : Optional[float]
        the final temperature value to use "..."
    xi: float
        the xi value for EI and PI metrics
    beta : float
        the beta value for the UCB metric
    stochastic_preds : bool
        whether the prediction values are generated through stochastic means
    threshold : float
        the threshold value to use in the random_threshold metric
    verbose : int
        the level of output the acquirer should print
    dim : int
        the number of objectives optimized
    pareto_front : Optional[Pareto]
        an object that stores and updates the Pareto front
        only defined if self.dim > 1

    Parameters
    ----------
    size : int
    init_size : Union[int, float] (Default = 0.01)
        the number of ligands or fraction of the pool to acquire initially.
    batch_sizes : Iterable[Union[int, float]] (Default = [0.01])
        the number of inputs or fraction of the pool to acquire in each
        successive batch. Will successively use each value in the list after
        each call to acquire_batch(), repeating the final value as necessary.
    metric : str (Default = 'greedy')
    epsilon : float (Default = 0.)
    temp_i : Optional[float] (Default = None)
    temp_f : Optional[float] (Default = 1.)
    xi: float (Default = 0.01)
    beta : int (Default = 2)
    threshold : float (Default = float('-inf'))
    seed : Optional[int] (Default = None)
        the random seed to use for initial batch acquisition
    verbose : int (Default = 0)
    dim : int (Default = 1)
    **kwargs
        additional and unused keyword arguments
    """
    def __init__(self, size: int,
                 init_size: Union[int, float] = 0.01,
                 batch_sizes: Iterable[Union[int, float]] = [0.01],
                 metric: str = 'greedy', dim: int = 1,
                 nadir: Iterable[float] = (0),
                 epsilon: float = 0., beta: int = 2, xi: float = 0.01,
                 threshold: float = float('-inf'),
                 stochastic_preds: bool = False,
                 temp_i: Optional[float] = None, temp_f: Optional[float] = 1.,
                 seed: Optional[int] = None, verbose: int = 0, **kwargs):
        self.size = size
        self.init_size = init_size
        self.batch_sizes = batch_sizes

        self.metric = metric
        self.dim = dim
        self.pareto_front = Pareto(num_objectives=self.dim)

        self.nadir = np.array(nadir, dtype=float)

        self.stochastic_preds = stochastic_preds

        if not 0. <= epsilon <= 1.:
            raise ValueError(f'Epsilon(={epsilon}) must be in [0, 1]')
        self.epsilon = epsilon

        self.beta = beta
        self.xi = xi
        self.threshold = threshold

        self.temp_i = temp_i
        self.temp_f = temp_f

        metrics.set_seed(seed)
        self.verbose = verbose

    def __len__(self) -> int:
        return self.size

    @property
    def dim(self) -> int:
        """The dimension of the objective for which we are acquiring"""
        return self.__dim

    @dim.setter
    def dim(self, dim):
        if dim < 1:
            raise ValueError(f'Invalid dimension for acquisition. got: {dim}')

        self.__dim = dim

    @property
    def needs(self) -> Set[str]:
        """Set[str] : the values this acquirer needs to calculate acquisition
        utilities"""
        return metrics.get_needs(self.metric)

    @property
    def init_size(self) -> int:
        """the number of inputs to acquire initially"""
        return self.__init_size

    @init_size.setter
    def init_size(self, init_size: Union[int, float]):
        if isinstance(init_size, float):
            if init_size < 0 or init_size > 1:
                raise ValueError(f'init_size(={init_size} must be in [0, 1]')
            init_size = math.ceil(self.size * init_size)
        if init_size < 0:
            raise ValueError(f'init_size(={init_size}) must be positive')

        self.__init_size = init_size

    @property
    def batch_sizes(self) -> List[int]:
        """the number of inputs to acquire in exploration batch"""
        return self.__batch_sizes

    @batch_sizes.setter
    def batch_sizes(self, batch_sizes: Iterable[Union[int, float]]):
        self.__batch_sizes = [bs for bs in batch_sizes]

        for i, bs in enumerate(self.__batch_sizes):
            if isinstance(bs, float):
                if bs < 0 or bs > 1:
                    raise ValueError(f'batch_size(={bs} must be in [0, 1]')
                self.__batch_sizes[i] = math.ceil(self.size * bs)
            if bs < 0:
                raise ValueError(f'batch_size(={bs} must be positive')

    def batch_size(self, t: int):
        """return the batch size to use for iteration t"""
        try:
            return self.batch_sizes[t]
        except IndexError:
            return self.batch_sizes[-1]

    def acquire_initial(self, xs: Iterable[T],
                        cluster_ids: Optional[Iterable[int]] = None,
                        cluster_sizes: Optional[Mapping[int, int]] = None,
                        ) -> List[T]:
        """Acquire an initial set of inputs to explore

        Parameters
        ----------
        xs : Iterable[T]
            an iterable of the inputs to acquire
        size : int
            the size of the iterable
        cluster_ids : Optional[Iterable[int]] (Default = None)
            a parallel iterable for the cluster ID of each input
        cluster_sizes : Optional[Mapping[int, int]] (Default = None)
            a mapping from a cluster id to the sizes of that cluster

        Returns
        -------
        List[T]
            the list of inputs to explore
        """
        U = metrics.random(np.empty(self.size))

        if cluster_ids is None and cluster_sizes is None:
            heap = []
            for x, u in tqdm(zip(xs, U), total=U.size, desc='Acquiring'):
                if len(heap) < self.init_size:
                    heapq.heappush(heap, (u, x))
                else:
                    heapq.heappushpop(heap, (u, x))
        else:
            d_cid_heap = {
                cid: ([], math.ceil(self.init_size * cluster_size/U.size))
                for cid, cluster_size in cluster_sizes.items()
            }

            for x, u, cid in tqdm(zip(xs, U, cluster_ids), total=U.size,
                                  desc='Acquiring'):
                heap, heap_size = d_cid_heap[cid]
                if len(heap) < heap_size:
                    heapq.heappush(heap, (u, x))
                else:
                    heapq.heappushpop(heap, (u, x))

            heaps = [heap for heap, _ in d_cid_heap.values()]
            heap = list(chain(*heaps))

        if self.verbose > 0:
            print(f'  Selected {len(heap)} initial samples')

        return [x for _, x in heap]

    def acquire_batch(self, xs: Iterable[T],
                      y_means: Iterable[float], y_vars: Iterable[float],
                      explored: Optional[Mapping] = None, k: int = 1,
                      cluster_ids: Optional[Iterable[int]] = None,
                      cluster_sizes: Optional[Mapping[int, int]] = None,
                      cluster_superset: Optional[bool] = None,
                      cluster_type: Optional[str] = None,
                      objective: Optional[Union[Objective, MultiObjective]] = None,
                      featurizer: Optional[Featurizer] = None, 
                      t: Optional[int] = None, **kwargs) -> List[T]:
        """Acquire a batch of inputs to explore

        Parameters
        ----------
        xs : Iterable[T]
            an iterable of the inputs to acquire
        y_means : Iterable[float]
            the predicted input values
        y_vars : Iterable[float]
            the variances of the predicted input values
        explored : Mapping[T, float] (Default = {})
            the set of explored inputs and their associated scores
        k : int, default=1
            the number of top-scoring compounds we are searching for. By
            default, assume we're looking for only the top-1 compound
        cluster_ids : Optional[Iterable[int]] (Default = None)
            a parallel iterable for the cluster ID of each input
        cluster_sizes : Optional[Mapping[int, int]] (Default = None)
            a mapping from a cluster id to the sizes of that cluster
        cluster_superset: Optional[int] (Default = None)
            number of molecules in subset to cluster into #batch_size clusters
            based on features and acquire cluster medoid 
            If not None, requires cluster_type and either featurizer or objective
        cluster_type: Optional[str] (Default = None)
            'fps': clusters according to fingerprints (requires featurizer input)
            'objs': clusters in objective space (requires objective input)
        objective: Optional[Union[Objective, MultiObjective]]
            objective to use if clustering in the objective space 
            required if clister_superset not None and cluster_type = 'objs'
        featurizer: Optional[Featurizer] (Default = None)
            featurizer to use for clustering if cluster_superset not None 
            and cluster_type = 'fps'
        t : Optional[int] (Default = None)
            the current iteration of batch acquisition
        is_random : bool (Default = False)
            are the y_means generated through stochastic methods?

        Returns
        -------
        List[T]
            a list of selected inputs
        """
        current_max = -np.inf
        P_f = np.array([current_max] * self.dim)

        if explored:
            ys = list(explored.values())
            Y = np.nan_to_num(np.array(ys, dtype=float), nan=-np.inf)
            if self.dim == 1:
                current_max = np.partition(Y, -k)[-k]
            else:
                P_f = Acquirer.pareto_frontier(Y)
                self.pareto_front.update_front(Y)
        else:
            explored = {}

        batch_size = self.batch_size(t)

        begin = default_timer()

        Y_means = np.array(y_means)
        Y_vars = np.array(y_vars)

        if self.verbose > 1:
            print('Calculating acquisition utilities ...', end=' ')

        U = metrics.calc(
            self.metric, Y_means=Y_means, Y_vars=Y_vars,
            pareto_front=self.pareto_front, current_max=current_max,
            threshold=self.threshold, beta=self.beta, xi=self.xi,
            stochastic=self.stochastic_preds, nadir=self.nadir
        )

        idxs = np.random.choice(
            np.arange(U.size), replace=False,
            size=int(batch_size * self.epsilon)
        )
        U[idxs] = np.inf

        if self.verbose > 1:
            print('Done!')
        if self.verbose > 2:
            total = default_timer() - begin
            mins, secs = divmod(int(total), 60)
            print(f'      Utility calculation took {mins}m {secs}s')

        if cluster_ids is None and cluster_sizes is None:
            heap = []
            for x, u in tqdm(zip(xs, U), total=U.size, desc='Acquiring'):
                if x in explored:
                    continue

                if len(heap) < batch_size:
                    heapq.heappush(heap, (u, x))
                else:
                    heapq.heappushpop(heap, (u, x))
        elif cluster_superset:
            superset = []
            for x, u in tqdm(zip(xs, U), total=U.size, desc='Acquiring superset'):
                if x in explored:
                    continue

                if len(superset) < cluster_superset:
                    heapq.heappush(superset, (u, x))
                else:
                    heapq.heappushpop(superset, (u, x))
            
            
            superset_xs = [x for _, x in superset]
            superset_us = [u for u, _ in superset]
            
            if cluster_type=='objs':
                cluster_basis = list(objective(superset_xs).values())
            elif cluster_type=='fps':
                cluster_basis = feature_matrix(superset_xs, featurizer=featurizer)

            cluster_ids = cluster_fps(fps=cluster_basis, 
                ncluster=self.batch_size, 
                method='kmeans', 
                ncpu=ray.cluster_resources()['CPU']
            )

            d_cid_heap = {cid: [] for cid in range(self.batch_size)}

            for x, u, cid in tqdm(zip(superset_xs, superset_us, cluster_ids),
                                  total=len(superset), desc='Acquiring'):
                
                if x in explored: 
                    continue

                heap = d_cid_heap[cid]

                if len(heap) < 1:
                    heapq.heappush(heap, (u, x))
                else:
                    heapq.heappushpop(heap, (u, x))

            heaps = [heap for heap, _ in d_cid_heap.values()]
            heap = list(chain(*heaps))

        else:
            # TODO(degraff): fix for epsilon-X approaches
            # the random indices are not distributed evenly amongst clusters
            # TODO(degraff): fix for MOO
            d_cid_heap = {
                cid: ([], math.ceil(batch_size * cluster_size/U.size))
                for cid, cluster_size in cluster_sizes.items()
            }

            global_pred_max = float('-inf')

            for x, y_pred, u, cid in tqdm(zip(xs, Y_means, U, cluster_ids),
                                          total=U.size, desc='Acquiring'):
                global_pred_max = max(y_pred, global_pred_max)

                if x in explored:
                    continue

                heap, heap_size = d_cid_heap[cid]
                if len(heap) < heap_size:
                    heapq.heappush(heap, (u, x))
                else:
                    heapq.heappushpop(heap, (u, x))

            if self.temp_i and self.temp_f:
                d_cid_heap = self._scale_heaps(
                    d_cid_heap, global_pred_max, t,
                    self.temp_i, self.temp_f
                )

            heaps = [heap for heap, _ in d_cid_heap.values()]
            heap = list(chain(*heaps))

        if self.verbose > 1:
            print(f'Selected {len(heap)} new samples')
        if self.verbose > 2:
            total = default_timer() - begin
            mins, secs = divmod(int(total), 60)
            print(f'      Batch acquisition took {mins}m {secs}s')

        return [x for _, x in heap]

    def _scale_heaps(self, d_cid_heap: Dict[int, List],
                     pred_global_max: float, t: int,
                     temp_i: float, temp_f: float):
        """Scale each heap's size based on a decay factor

        The decay factor is calculated by an exponential decay based on the
        difference between a given heap's local maximum and the predicted
        global maximum then scaled by the current temperature. The temperature
        is also an exponential decay based on the current iteration starting at
        the initial temperature and approaching the final temperature.

        Parameters
        ----------
        d_cid_heap : Dict[int, List]
            a mapping from cluster_id to the heap containing the inputs to
            acquire from that cluster
        pred_global_max : float
            the predicted maximum value of the objective function
        t : int
            the current iteration of acquisition
        temp_i : float
            the initial temperature of the system
        temp_f : float
            the final temperature of the system

        Returns
        -------
        d_cid_heap
            the original mapping scaled down by the calculated decay factor
        """
        temp = self._calc_temp(t, temp_i, temp_f)

        for cid, (heap, heap_size) in d_cid_heap.items():
            if len(heap) == 0:
                continue

            pred_local_max = max(
                heap, key=lambda yx: -1 if math.isinf(yx[0]) else yx[0]
            )
            f = self._calc_decay(pred_global_max, pred_local_max, temp)

            new_heap_size = math.ceil(f * heap_size)
            new_heap = heapq.nlargest(new_heap_size, heap)

            d_cid_heap[cid] = (new_heap, new_heap_size)

        return d_cid_heap

    @classmethod
    def _calc_temp(cls, t: int, temp_i, temp_f) -> float:
        """Calculate the temperature of the system"""
        return temp_i * math.exp(-t/0.75) + temp_f

    @classmethod
    def _calc_decay(cls, global_max: float, local_max: float,
                    temp: float) -> float:
        """Calculate the decay factor of a given heap"""
        return math.exp(-(global_max - local_max)/temp)

    @staticmethod
    def pareto_frontier(Y: np.ndarray) -> np.ndarray:
        """calculate the pareto frontier for the objective matrix Y

        code from: https://github.com/QUVA-Lab/artemis/
                   blob/peter/artemis/general/pareto_efficiency.py"""
        efficient_idxs = np.arange(Y.shape[0])
        Y_copy = Y
        idx = 0
        while idx < len(Y_copy):
            nondominated_point_mask = np.any(Y_copy > Y_copy[idx], axis=1)
            nondominated_point_mask[idx] = True

            efficient_idxs = efficient_idxs[nondominated_point_mask]
            Y_copy = Y_copy[nondominated_point_mask]

            idx = np.sum(nondominated_point_mask[:idx]) + 1

        return Y[efficient_idxs]
        # if return_mask:
        #     is_efficient_mask = np.zeros(N, dtype = bool)
        #     is_efficient_mask[is_efficient] = True
        #     return is_efficient_mask
        # else:
        #     return is_efficient

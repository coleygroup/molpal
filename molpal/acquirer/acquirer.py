"""This module contains the Acquirer class, which is used to gather inputs for
a subsequent round of exploration based on prior prediction data."""

import heapq
from itertools import chain, zip_longest
import math
import random
from timeit import default_timer
from typing import Dict, Iterable, List, Mapping, Optional, Set, TypeVar, Union

import numpy as np
from tqdm import tqdm

from . import metrics

T = TypeVar('T')

class Acquirer:
    """An Acquirer acquires inputs from an input pool for exploration.

    Attributes
    ----------
    metric : Callable[..., np.ndarray]
        the function for calculating the acquisition utility of an input
    metric_type : str
        the alias of the metric function (NOT the function name)
    needs : Set[str]
        the values this acquirer needs to calculate acquisition utilities
    init_size : int
        the number of inputs to acquire initially
    batch_size : int
        the number of inputs to acquire in each batch
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
    seed : Optional[int]
        the random seed to use for initial batch acquisition
    verbose : int
        the level of output the acquirer should print
    
    Parameters
    ----------
    pool_size : int
        the size of the pool on which this acquirer will operate
    metric : str (Default = 'greedy')
        the alias of the metric to use
    init_size : Union[int, float] (Default = 0.02)
        the number of ligands or fraction of the pool to acquire initially.
    batch_size : Union[int, float] (Default = 0.02)
        the number of ligands or fraction of the pool to acquire
        in each batch
    """
    def __init__(self, size: int,
                 init_size: Union[int, float] = 0.01,
                 batch_size: Union[int, float] = 0.01,
                 metric: str = 'greedy',
                 epsilon: float = 0., beta: int = 2, xi: float = 0.01,
                 threshold: float = float('-inf'),
                 temp_i: Optional[float] = None, temp_f: Optional[float] = 1.,
                 seed: Optional[int] = None, verbose: int = 0, **kwargs):
        self.size = size
        self.init_size = init_size
        self.batch_size = batch_size

        self.metric = metric
        self.stochastic_preds = False

        if not 0. <= epsilon <= 1.:
            raise ValueError(f'Epsilon(={epsilon}) must be in [0, 1]')
        self.epsilon = epsilon
        
        self.beta = beta
        self.xi = xi
        self.threshold = threshold

        self.temp_i = temp_i
        self.temp_f = temp_f

        metrics.seed(seed)
        self.verbose = verbose

    def __len__(self) -> int:
        return self.size

    @property
    def metric(self):
        return self.__metric

    @metric.setter
    def metric(self, metric: str):
        self.__metric_type = metric
        self.__metric = metrics.get_metric(metric)

    @property
    def metric_type(self) -> str:
        return self.__metric_type

    @property
    def needs(self) -> Set[str]:
        return metrics.get_needs(self.__metric_type)
    
    @property
    def init_size(self) -> int:
        if isinstance(self.__init_size, float):
            return math.ceil(self.size * self.__init_size)

        return self.__init_size
    
    @init_size.setter
    def init_size(self, init_size: Union[int ,float]):
        if isinstance(init_size, float) and (init_size<0 or init_size>1):
            raise ValueError(f'init_size(={init_size} must be in [0, 1]')
        elif init_size < 0:
            raise ValueError(f'init_size(={init_size}) must be positive')

        self.__init_size = init_size

    @property
    def batch_size(self) -> int:
        if isinstance(self.__batch_size, float):
            return math.ceil(self.size * self.__batch_size)

        return self.__batch_size
    
    @batch_size.setter
    def batch_size(self, batch_size):
        if isinstance(batch_size, float) and (batch_size<0 or batch_size>1):
            raise ValueError(f'batch_size(={batch_size} must be in [0, 1]')
        elif batch_size < 0:
            raise ValueError(f'batch_size(={batch_size} must be positive')

        self.__batch_size = batch_size

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
        U = metrics.random_metric(np.empty(self.size))

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
                      explored: Mapping[T, float],
                      cluster_ids: Optional[Iterable[int]] = None,
                      cluster_sizes: Optional[Mapping[int, int]] = None,
                      epoch: Optional[int] = None, is_random: bool = False,
                      **kwargs) -> List[T]:
        """Acquire a batch of inputs to explore

        Parameters
        ----------
        xs : Iterable[T]
            an iterable of the inputs to acquire
        y_means : Iterable[float]
            the predicted input values
        y_vars : Iterable[float]
            the variances of the predicted input values
        explored : Mapping[T, float]
            the set of explored inputs and their associated scores
        cluster_ids : Optional[Iterable[int]] (Default = None)
            a parallel iterable for the cluster ID of each input
        cluster_sizes : Optional[Mapping[int, int]] (Default = None)
            a mapping from a cluster id to the sizes of that cluster
        epoch : Optional[int] (Default = None)
            the current epoch of batch acquisition
        is_random : bool (Default = False)
            are the y_means generated through stochastic methods?

        Returns
        -------
        List[T]
            a list of selected inputs
        """
        current_max = max(y for y in explored.values() if y is not None)

        begin = default_timer()

        Y_mean = np.array(y_means)
        Y_var = np.array(y_vars)

        if self.verbose > 1:
            print('Calculating acquisition utilities ...', end=' ')

        U = self.metric(Y_mean=Y_mean, Y_var=Y_var, current_max=current_max,
                        beta=self.beta, xi=self.xi, threshold=self.threshold,
                        stochastic=self.stochastic_preds)

        idxs = np.random.choice(
            np.arange(U.size), replace=False,
            size=int(self.batch_size * U.size * self.epsilon)
        )
        U[idxs] = np.inf

        if self.verbose > 1:
            print('Done!')
        if self.verbose > 2:
            total = default_timer() - begin
            m, s = divmod(int(total), 60)
            print(f'      Utility calculation took {m}m {s}s')
        
        if cluster_ids is None and cluster_sizes is None:
            # unclustered acquisition maintains one heap that pushes
            # inputs on based on their utility
            heap = []
            for x, u in tqdm(zip(xs, U), total=U.size, desc='Acquiring'):
                if x in explored:
                    continue

                if len(heap) < self.batch_size:
                    heapq.heappush(heap, (u, x))
                else:
                    heapq.heappushpop(heap, (u, x))
        else:
            # this is broken for e-greedy/pi/etc. approaches
            # the random indices are not distributed evenly amongst clusters

            # clustered acquisition maintains m independent heaps that
            # operate similarly to the above
            d_cid_heap = {
                cid: ([], math.ceil(self.batch_size * cluster_size/U.size))
                for cid, cluster_size in cluster_sizes.items()
            }

            global_pred_max = float('-inf')

            for x, y_pred, u, cid in tqdm(zip(xs, Y_mean, U, cluster_ids),
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
                    d_cid_heap, global_pred_max, epoch, 
                    self.temp_i, self.temp_f
                )

            heaps = [heap for heap, _ in d_cid_heap.values()]
            heap = list(chain(*heaps))

        if self.verbose > 1:
            print(f'Selected {len(heap)} new samples')
        if self.verbose > 2:
            total = default_timer() - begin
            m, s = divmod(int(total), 60)
            print(f'      Batch acquisition took {m}m {s}s')

        return [x for _, x in heap]

    def _scale_heaps(self, d_cid_heap: Dict[int, List], 
                     pred_global_max: float, epoch: int, 
                     temp_i: float, temp_f: float):
        """Scale each heap's size based on a decay factor

        The decay factor is calculated by an exponential decay based on the
        difference between a given heap's local maximum and the predicted 
        global maximum then scaled by the current temperature. The temperature 
        is in turn an exponential decay based on the current epoch starting at 
        the initial temperature and approaching the final temperature.

        Parameters
        ----------
        d_cid_heap : Dict[int, List]
            a mapping from cluster_id to the heap containing the inputs to
            acquire from that cluster
        pred_global_max : float
            the predicted maximum value of the objective function
        epoch : int
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
        temp = self._calc_temp(epoch, temp_i, temp_f)

        for cid, (heap, heap_size) in d_cid_heap.items():
            if len(heap) == 0:
                continue
            
            # get the largest non-infinity value
            pred_local_max = max(
                heap, key=lambda yx: -1 if math.isinf(yx[0]) else yx[0]
            )
            f = self._calc_decay(pred_global_max, pred_local_max, temp)

            new_heap_size = math.ceil(f * heap_size)
            new_heap = heapq.nlargest(new_heap_size, heap)

            d_cid_heap[cid] = (new_heap, new_heap_size)

        return d_cid_heap

    @classmethod
    def _calc_temp(cls, epoch: int, temp_i, temp_f) -> float:
        """Calculate the temperature of the system"""
        return temp_i * math.exp(-epoch/0.75) + temp_f

    @classmethod
    def _calc_decay(cls, global_max: float, local_max: float,
                    temp: float) -> float:
        """Calculate the decay factor of a given heap"""
        return math.exp(-(global_max - local_max)/temp)


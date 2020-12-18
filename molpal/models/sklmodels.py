from functools import partial
import logging
from pathlib import Path
import pickle
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from tqdm import tqdm

from molpal.models.base import Model
from molpal.models.utils import batches, feature_matrix

T = TypeVar('T')

class RFModel(Model):
    """A Random Forest model ensemble for estimating mean and variance

    Attributes (instance)
    ----------
    n_jobs : int
        the number of jobs to parallelize training and prediction over
    model : RandomForestRegressor
        the underlying model on which to train and perform inference
    
    Parameters
    ----------
    test_batch_size : Optional[int] (Default = 10000)
        the size into which testing data should be batched
    ncpu : int (Default = 1)
        the number of cores training/inference should be distributed over
    """
<<<<<<< HEAD
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 8,
                 min_samples_leaf: int = 1,
                 test_batch_size: Optional[int] = 4096,
=======

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 8,
                 min_samples_leaf=1, test_batch_size: Optional[int] = 4096,
>>>>>>> 50998cb5548c13f0bed3ce0ed70301b9364374af
                 num_workers: int = 1, ncpu: int = 1,
                 distributed: bool = False, **kwargs):
        test_batch_size = test_batch_size or 4096

        if distributed:
            from mpi4py import MPI
<<<<<<< HEAD

=======
>>>>>>> 50998cb5548c13f0bed3ce0ed70301b9364374af
            num_workers = MPI.COMM_WORLD.Get_size()
            n_jobs = ncpu

            # if num_workers > 2:
            #     test_batch_size *= num_workers
        else:
            n_jobs = ncpu * num_workers

<<<<<<< HEAD
        self.ncpu = ncpu

=======
>>>>>>> 50998cb5548c13f0bed3ce0ed70301b9364374af
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
<<<<<<< HEAD
            verbose=0,
=======
>>>>>>> 50998cb5548c13f0bed3ce0ed70301b9364374af
        )

        super().__init__(test_batch_size, num_workers=num_workers,
                         distributed=distributed, **kwargs)

    @property
    def provides(self):
        return {'means', 'vars'}

    @property
    def type_(self):
        return 'rf'

    def train(self, xs: Iterable[T], ys: Iterable[float], *,
              featurize: Callable[[T], ndarray], retrain: bool = True):
        # retrain means nothing for this model- internally it always retrains
        X = feature_matrix(xs, featurize,
                           self.num_workers, self.ncpu, self.distributed)
        Y = np.array(ys)

        self.model.fit(X, Y)
        Y_pred = self.model.predict(X)
        errors = Y_pred - Y
        logging.info(f'  training MAE: {np.mean(np.abs(errors)):.2f},'
                     f'MSE: {np.mean(np.power(errors, 2)):.2f}')
        return True

    def get_means(self, xs: Sequence) -> ndarray:
        # this is only marginally faster
        # if self.distributed and self.num_workers > 2:
<<<<<<< HEAD
        #     predict_ = partial(predict, model=self.model)
        #     with self.MPIPool(max_workers=self.num_workers) as pool:
        #         xs_batches = batches(xs, self.test_batch_size//self.num_workers)
        #         Y = list(pool.map(predict_, xs_batches))
        #         return np.hstack(Y)
=======
        #     predict_ = partial(predict, xs=xs)
        #     with self.MPIPool(max_workers=self.num_workers) as pool:
        #         # xs_batches = batches(xs, self.test_batch_size//self.num_workers)
        #         # Y = list(pool.map(predict_, xs_batches))
        #         # return np.hstack(Y)
        #         Y = list(pool.map(predict_, self.model.estimators_))
        #     Y = np.vstack(Y).mean(axis=0)
>>>>>>> 50998cb5548c13f0bed3ce0ed70301b9364374af

        X = np.vstack(xs)
        return self.model.predict(X)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[ndarray, ndarray]:
        X = np.vstack(xs)
        preds = np.zeros((len(X), len(self.model.estimators_)))
        for j, submodel in enumerate(self.model.estimators_):
            preds[:, j] = submodel.predict(xs)

        return np.mean(preds, axis=1), np.var(preds, axis=1)

<<<<<<< HEAD
# def predict(xs, model):
#     X = np.vstack(xs)
#     return model.predict(X)
=======
    def apply(self, *, x_feats: Iterable, size: Optional[int] = None,
              mean_only: bool = True,
              **kwargs) -> Tuple[List[float], List[float]]:
        n_batches = (size//self.test_batch_size) + 1 if size else None
        xs = batches(x_feats, self.test_batch_size)

        means = []
        variances = []

        if self.distributed and self.num_workers > 2 and mean_only:
            from mpi4py.futures import MPIPoolExecutor

            predict_ = partial(predict, model=self.model)
            with MPIPoolExecutor(max_workers=self.num_workers) as pool:
                Y = list(pool.map(predict_, xs))
                return np.hstack(Y)

        if mean_only:
            for batch_xs in tqdm(xs, total=n_batches, smoothing=0.,
                                 desc='Inference', unit='batch'):
                batch_means = self.get_means(batch_xs)
                means.extend(batch_means)
        else:
            for batch_xs in tqdm(xs, total=n_batches, smoothing=0.,
                                 desc='Inference', unit='batch'):
                batch_means, batch_vars = self.get_means_and_vars(batch_xs)
                means.extend(batch_means)
                variances.extend(batch_vars)

        return means, variances

def predict(xs, model):
    X = np.vstack(xs)
    return model.predict(X)
>>>>>>> 50998cb5548c13f0bed3ce0ed70301b9364374af

class GPModel(Model):
    """Gaussian process model
    
    Attributes
    ----------
    model : GaussianProcessRegressor
    kernel : kernels.Kernel
        the GP kernel that will be used

    Parameters
    ----------
    gp_kernel : str (Default = 'dotproduct')
    ncpu : int (Default = 0)
    test_batch_size : Optional[int] (Default = 1000)
    """
    def __init__(self, gp_kernel: str = 'dotproduct',
                 test_batch_size: Optional[int] = 1024,
                 num_workers: int = 1, ncpu: int = 1,
                 distributed: bool = False, **kwargs):
        test_batch_size = test_batch_size or 1024

        self.ncpu = ncpu

        self.model = None
        self.kernel = {
            'dotproduct': kernels.DotProduct
        }[gp_kernel]()

        super().__init__(test_batch_size, num_workers=num_workers,
                         distributed=distributed, **kwargs)
        
    @property
    def provides(self):
        return {'means', 'vars'}

    @property
    def type_(self):
        return 'gp'

    def train(self, xs: Iterable[T], ys: Iterable[float], *,
              featurize: Callable[[T], ndarray], retrain: bool = False) -> bool:
        X = feature_matrix(xs, featurize,
                           self.num_workers, self.ncpu, self.distributed)
        Y = np.array(ys)

        self.model = GaussianProcessRegressor(kernel=self.kernel)
        self.model.fit(X, Y)
        Y_pred = self.model.predict(X)
        errors = Y_pred - Y
        logging.info('  training MAE: {:.2f}, MSE: {:.2f}'.format(
            np.mean(np.abs(errors)), np.mean(np.power(errors, 2))
        ))
        return True

    def get_means(self, xs: Sequence) -> ndarray:
        X = np.vstack(xs)

        return self.model.predict(X)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[ndarray, ndarray]:
        X = np.vstack(xs)
        Y_mean, Y_sd = self.model.predict(X, return_std=True)

        return Y_mean, np.power(Y_sd, 2)
        
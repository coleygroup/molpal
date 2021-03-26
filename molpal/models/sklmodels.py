from functools import partial
import logging
from pathlib import Path
import pickle
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar

import joblib
import numpy as np
from numpy import ndarray
import ray
from ray.util.joblib import register_ray
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from tqdm import tqdm

from molpal.encoder import feature_matrix
from molpal.models.base import Model

T = TypeVar('T')

register_ray()

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
    """
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 8,
                 min_samples_leaf: int = 1,
                 test_batch_size: Optional[int] = 65536,
                 **kwargs):
        test_batch_size = test_batch_size or 65536

        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, n_jobs=-1,
        )

        super().__init__(test_batch_size, **kwargs)

    @property
    def provides(self):
        return {'means', 'vars'}

    @property
    def type_(self):
        return 'rf'

    def train(self, xs: Iterable[T], ys: Iterable[float], *,
              featurizer: Callable[[T], ndarray], retrain: bool = True):
        # retrain means nothing for this model- internally it always retrains
        X = feature_matrix(xs, featurizer)
        Y = np.array(ys)

        with joblib.parallel_backend('ray'):
            self.model.fit(X, Y)
            Y_pred = self.model.predict(X)

        errors = Y_pred - Y
        logging.info(f'  training MAE: {np.mean(np.abs(errors)):.2f},'
                     f'MSE: {np.mean(np.power(errors, 2)):.2f}')
        return True

    def get_means(self, xs: Sequence) -> ndarray:
        X = np.vstack(xs)
        with joblib.parallel_backend('ray'):
            return self.model.predict(X)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[ndarray, ndarray]:
        X = np.vstack(xs)
        preds = np.zeros((len(X), len(self.model.estimators_)))

        with joblib.parallel_backend('ray'):
            for j, submodel in enumerate(self.model.estimators_):
                preds[:, j] = submodel.predict(xs)

        return np.mean(preds, axis=1), np.var(preds, axis=1)

# def predict(xs, model):
#     X = np.vstack(xs)
#     return model.predict(X)

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
    test_batch_size : Optional[int] (Default = 1000)
    """
    def __init__(self, gp_kernel: str = 'dotproduct',
                 test_batch_size: Optional[int] = 1024,
                 **kwargs):
        test_batch_size = test_batch_size or 1024

        self.model = None
        self.kernel = {
            'dotproduct': kernels.DotProduct
        }[gp_kernel]()

        super().__init__(test_batch_size, **kwargs)
        
    @property
    def provides(self):
        return {'means', 'vars'}

    @property
    def type_(self):
        return 'gp'

    def train(self, xs: Iterable[T], ys: Iterable[float], *,
              featurizer, retrain: bool = False) -> bool:
        X = feature_matrix(xs, featurizer)
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
        
import logging
from pathlib import Path
import pickle
from typing import Callable, Iterable, Optional, Sequence, Tuple, TypeVar

import numpy as np
from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from .base import Model
from .utils import feature_matrix

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
    njobs : int (Default = -1)
        the number of jobs training/inference should be distributed over
    """

    def __init__(self, test_batch_size: Optional[int] = 10000,
                 njobs: int = -1, **kwargs):
        test_batch_size = test_batch_size or 10000
        self.n_jobs = njobs

        self.model = RandomForestRegressor(
            n_estimators=100,
            n_jobs=self.n_jobs,
            max_depth=8,
            verbose=1,
        )

        super().__init__(test_batch_size, **kwargs)

    @property
    def provides(self):
        return {'means', 'vars'}

    @property
    def type_(self):
        return 'rf'

    def train(self, xs: Iterable[T], ys: Iterable[float], *,
              featurize: Callable[[T], ndarray], retrain: bool = True):
        # retrain means nothing for this model- internally it always retrains
        X = feature_matrix(xs, featurize, 1)#self.n_jobs)
        Y = np.array(ys)

        self.model.fit(X, Y)
        Y_pred = self.model.predict(X)
        errors = Y_pred - Y
        logging.info('  training MAE: {:.2f}, MSE: {:.2f}'.format(
            np.mean(np.abs(errors)), np.mean(np.power(errors, 2))
        ))
        return True

    def get_means(self, xs: Sequence) -> ndarray:
        X = np.stack(xs, axis=0)
        return self.model.predict(X)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[ndarray, ndarray]:
        X = np.stack(xs, axis=0)
        preds = np.zeros((len(X), len(self.model.estimators_)))
        for j, submodel in enumerate(self.model.estimators_):
            preds[:, j] = submodel.predict(xs)

        return np.mean(preds, axis=1), np.var(preds, axis=1)

    def save(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

        model_path = f'{path}/model.pkl'
        pickle.dump(self.model, open(model_path, 'wb'))
    
    def load(self, path) -> None:
        model_path = f'{path}/model.pkl'
        self.model = pickle.load(open(model_path, 'wb'))
    
class GPModel(Model):
    """Gaussian process model
    
    Attributes
    ----------
    model : GaussianProcessRegressor
    kernel : kernels.Kernel
        the GP kernel that will be used
    njobs : int
    
    Parameters
    ----------
    gp_kernel : str (Default = 'dotproduct')
        the type of GP kernel to use
    njobs : int (Default = 0)
        the number of jobs to parallelize feature matrix generation over
    test_batch_size : Optional[int] (Default = 1000)
        the size into which testing data should be batched
    """
    def __init__(self, gp_kernel: str = 'dotproduct', njobs: int = 0,
                 test_batch_size: Optional[int] = 1000, **kwargs):
        test_batch_size = test_batch_size or 1000
        self.model = None
        self.kernel = {
            'dotproduct': kernels.DotProduct
        }[gp_kernel]()
        self.njobs = njobs

        super().__init__(test_batch_size, **kwargs)

    @property
    def provides(self):
        return {'means', 'vars'}

    @property
    def type_(self):
        return 'gp'

    def train(self, xs: Iterable[T], ys: Iterable[float], *,
              featurize: Callable[[T], ndarray], retrain: bool = False) -> bool:
        X = feature_matrix(xs, featurize, 1)#self.njobs)
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
        X = np.stack(xs, axis=0)

        return self.model.predict(X)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[ndarray, ndarray]:
        X = np.stack(xs, axis=0)
        Y_mean, Y_sd = self.model.predict(X, return_std=True)

        return Y_mean, np.power(Y_sd, 2)

    def save(self, path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

        model_path = f'{path}/model.pkl'
        pickle.dump(self.model, open(model_path, 'wb'))
    
    def load(self, path) -> None:
        model_path = f'{path}/model.pkl'
        self.model = pickle.load(open(model_path, 'wb'))
        
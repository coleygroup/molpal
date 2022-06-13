"""This module contains Model implementations that utilize the sklearn models
as their underlying model"""
import logging
from pathlib import Path
import pickle
from typing import Callable, Iterable, Optional, Sequence, Tuple, TypeVar

import joblib
import numpy as np
import ray
from ray.util.joblib import register_ray
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from molpal.featurizer import feature_matrix
from molpal.models.base import Model

T = TypeVar("T")

register_ray()

logger = logging.getLogger(__name__)


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
    test_batch_size : Optional[int], default=32768
        the size into which testing data should be batched
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 8,
        min_samples_leaf: int = 1,
        test_batch_size: Optional[int] = 32768,
        model_seed: Optional[int] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 32768

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=model_seed,
        )

        super().__init__(test_batch_size, **kwargs)

    @property
    def provides(self):
        return {"means", "vars"}

    @property
    def type_(self):
        return "rf"

    def train(
        self,
        xs: Iterable[T],
        ys: Iterable[float],
        *,
        featurizer: Callable[[T], np.ndarray],
        retrain: bool = True,
    ):
        X = np.array(feature_matrix(xs, featurizer))
        Y = np.array(ys)

        with joblib.parallel_backend("ray"):
            self.model.fit(X, Y)
            Y_pred = self.model.predict(X)

        errors = Y_pred - Y
        logger.info(
            f"training MAE: {np.abs(errors).mean():0.2f}," f"MSE: {(errors**2).mean():0.2f}"
        )
        return True

    def get_means(self, xs: Sequence) -> np.ndarray:
        X = np.vstack(xs)
        with joblib.parallel_backend("ray"):
            return self.model.predict(X)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack(xs)

        X = ray.put(X)
        trees = [ray.put(tree) for tree in self.model.estimators_]
        refs = [RFModel.subtree_predict.remote(tree, X) for tree in trees]
        predss = np.array(ray.get(refs))

        return np.mean(predss, 0), np.var(predss, 0)

    def save(self, path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_path = str(path / "model.pkl")
        pickle.dump(self.model, open(model_path, "wb"))

        return model_path

    def load(self, path):
        self.model = pickle.load(open(path, "rb"))

    @staticmethod
    @ray.remote
    def subtree_predict(tree, xs):
        return tree.predict(xs)


class GPModel(Model):
    """Gaussian process model

    Attributes
    ----------
    model : GaussianProcessRegressor
    kernel : type[kernels.Kernel]
        the GP kernel that will be used

    Parameters
    ----------
    gp_kernel : str (Default = 'dot')
    test_batch_size : Optional[int] (Default = 1000)
    """

    def __init__(
        self,
        gp_kernel: str = "dot",
        test_batch_size: Optional[int] = 1024,
        model_seed: Optional[int] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 1024

        kernel = {"dot": kernels.DotProduct, "matern": kernels.Matern, "rbf": kernels.RBF}[
            gp_kernel
        ]()

        self.model = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=model_seed
        )
        super().__init__(test_batch_size, **kwargs)

    @property
    def provides(self):
        return {"means", "vars"}

    @property
    def type_(self):
        return "gp"

    def train(
        self, xs: Iterable[T], ys: Iterable[float], *, featurizer, retrain: bool = False
    ) -> bool:
        X = np.array(feature_matrix(xs, featurizer))
        Y = np.array(list(ys))

        self.model.fit(X, Y)
        Y_pred = self.model.predict(X)
        errors = Y_pred - Y
        logger.info(f"training MAE: {np.abs(errors).mean():0.2f}, MSE: {(errors**2).mean():0.2f}")
        return True

    def get_means(self, xs: Sequence) -> np.ndarray:
        X = np.vstack(xs)

        return self.model.predict(X)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack(xs)
        Y_mean_pred, Y_sd_pred = self.model.predict(X, return_std=True)

        return Y_mean_pred, Y_sd_pred**2

    def save(self, path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_path = str(path / "model.pkl")
        pickle.dump(self.model, open(model_path, "wb"))

        return model_path

    def load(self, path):
        self.model = pickle.load(open(path, "rb"))

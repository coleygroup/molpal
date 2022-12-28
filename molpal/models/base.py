"""This module contains the Model abstract base class. All custom models must
implement this interface in order to interact properly with an Explorer"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (Callable, Iterable, List, Optional,
                    Sequence, Set, Tuple, TypeVar)

import numpy as np
from tqdm import tqdm

from molpal.utils import batches

T = TypeVar('T')
T_feat = TypeVar('T_feat')


class MultiTaskModel:
    def __init__(self, models: Iterable[str], **kwargs) -> None:
        self.models = []
        for model in models:
            if model == 'rf':
                from molpal.models.sklmodels import RFModel
                m = RFModel(**kwargs)

            elif model == 'gp':
                from molpal.models.sklmodels import GPModel
                m = GPModel(**kwargs)

            elif model == 'nn':
                from molpal.models.nnmodels import nn
                m = nn(**kwargs)

            elif model == 'mpn':
                from molpal.models.mpnmodels import mpn
                m = mpn(**kwargs)

            elif model == 'random':
                from molpal.models.random import RandomModel
                m = RandomModel(**kwargs)

            else:
                raise NotImplementedError(f'Unrecognized model: "{model}"')

            self.models.append(m)

    def __len__(self):
        """the number of tasks this model predicts"""
        return len(self.models)
        
    @property
    @abstractmethod
    def provides(self) -> Set[str]:
        return set.intersection(*[model.provides for model in self.models])

    def train(
        self, xs: Iterable[T], yss: Sequence[Sequence[Optional[float]]],
        **kwargs
    ):
        Ys = np.array(yss).T
        for model, Y in zip(self.models, Ys):
            mask = ~np.isnan(Y)
            xs = np.array(xs)[mask]
            ys = Y[mask]
            model.train(xs, ys, **kwargs)

        return True

    # def apply(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    #     Y_preds = [model.apply(*args, **kwargs) for model in self.models]
    #     Y_preds_mean, Y_preds_var = zip(*Y_preds)

    #     return np.stack(Y_preds_mean, axis=1), np.stack(Y_preds_var, axis=1)

    def apply(
        self, x_ids: Iterable[T], x_feats: Iterable[T_feat],
        batched_size: Optional[int] = None,
        size: Optional[int] = None, mean_only: bool = True,
        disable: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the model to the inputs

        Parameters
        ----------
        x_ids : Iterable[T]
            an iterable of input identifiers that correspond to the
            uncompressed input representations
        x_feats : Iterable[T_feat]
            an iterable of either batches or individual uncompressed feature
            representations corresponding to the input identifiers
        batched_size : Optional[int] (Default = None)
            the size of the batches if xs is an iterable of batches
        size : Optional[int] (Default = None)
            the length of the iterable, if known
        mean_only : bool (Default = True)
            whether to generate the predicted variance in addition to the mean

        Returns
        -------
        means : np.ndarray
            the mean predicted values
        variances: np.ndarray
            the variance in the predicted means, empty if mean_only is True
        """
        if self.models[0].type_ == 'mpn':
            # ^^ ASSUMES BOTH MODELS ARE EITHER MPN OR NOT!!! TODO: change this 
            # MPNs predict directly on the input identifier
            xs = x_ids
            batched_size = None
        else:
            xs = x_feats

        if batched_size:
            n_batches = (size//batched_size) + 1 if size else None
        else:
            xs = batches(xs, self.models[0].test_batch_size)
            n_batches = (size//self.models[0].test_batch_size) + 1 if size else None
            batched_size = self.models[0].test_batch_size



        if mean_only:        
            meanss = []
            variancess = []
            for batch_xs in tqdm(
                xs, total=n_batches, desc='Inference', smoothing=0.,
                unit='smi', unit_scale=batched_size, disable=disable
            ):
                means = np.array([model.get_means(batch_xs)
                                 for model in self.models])
                meanss.append(means)
                variancess.append([])

            meansss = np.concatenate(meanss, axis=1).T
            variancesss = np.concatenate(variancess)
            return meansss, variancesss
        else:
            means_all = np.zeros((0,len(self)))
            vars_all = np.zeros((0,len(self)))
            for batch_xs in tqdm(
                xs, total=n_batches, desc='Inference', smoothing=0.,
                unit='smi', unit_scale=batched_size, disable=disable
            ):
                meanss = []
                varss = []
                for model in self.models: 
                    means, vars = model.get_means_and_vars(batch_xs)
                    meanss.append(means)
                    varss.append(vars)

                means_all = np.concatenate([means_all, np.array(meanss).T])
                vars_all = np.concatenate([vars_all, np.array(varss).T])

                # means = [model.get_means(batch_xs) for model in self.models]
                # variances = [model.get_means_and_vars(batch_xs)[1]
                #              for model in self.models]
                # meanss.append(means)
                # variancess.append(variances)


            # meansss = np.concatenate(meanss, axis=1).T
            # variancesss = np.concatenate(variancess, axis=1).T
            return means_all, vars_all

    def save(self, basepath: str) -> Iterable[str]:
        basepath = Path(basepath)
        return [
            model.save(basepath.with_name(f'{basepath.stem}_{i}'))
            for i, model in enumerate(self.models)
        ]
    
    def load(self, paths: Iterable[str]):
        for model, path in zip(self.models, paths):
            model.load(path)

class SingleTaskModel(ABC):
    """A Model can be trained on input data to predict the values for inputs
    that have not yet been evaluated.

    This is an abstract base class and cannot be instantiated by itself.

    Attributes
    ----------
    model(s)
        the model(s) used to calculate prediction values
    test_batch_size : int
        the size of the batch to split prediction inputs into if not
        already batched
    ncpu : int
        the total number of cores available to parallelize computation over
    additional, class-specific instance attributes

    Parameters
    ----------
    test_batch_size : int
    ncpu : int (Default = 1)
    """
    def __init__(self, test_batch_size: int, **kwargs):
        self.test_batch_size = test_batch_size

    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.apply(*args, **kwargs)

    @property
    @abstractmethod
    def provides(self) -> Set[str]:
        """The types of values this model provides"""

    @property
    @abstractmethod
    def type_(self) -> str:
        """The underlying architecture of the Model"""

    @abstractmethod
    def train(self, xs: Iterable[T], ys: Sequence[float], *,
              featurizer: Callable[[T], T_feat], retrain: bool = False) -> bool:
        """Train the model on the input data
        
        Parameters
        ----------
        xs : Iterable[T]
            an iterable of inputs in their identifier representation
        ys : Sequence[float]
            a parallel sequence of scalars that correspond to the regression
            target for each x
        featurize : Callable[[T], T_feat]
            a function that transforms an input from its identifier to its
            feature representation
        retrain : bool (Deafult = False)
            whether the model should be completely retrained
        """
        # TODO: hyperparameter optimizations in inner loop?

    @abstractmethod
    def get_means(self, xs: Sequence) -> np.ndarray:
        """Get the mean predicted values for a sequence of inputs"""

    @abstractmethod
    def get_means_and_vars(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        """Get both the predicted mean and variance for a sequence of inputs"""

    def apply(
        self, x_ids: Iterable[T], x_feats: Iterable[T_feat],
        batched_size: Optional[int] = None,
        size: Optional[int] = None, mean_only: bool = True, disable: bool = False 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the model to the inputs

        Parameters
        ----------
        x_ids : Iterable[T]
            an iterable of input identifiers that correspond to the
            uncompressed input representations
        x_feats : Iterable[T_feat]
            an iterable of either batches or individual uncompressed feature
            representations corresponding to the input identifiers
        batched_size : Optional[int] (Default = None)
            the size of the batches if xs is an iterable of batches
        size : Optional[int] (Default = None)
            the length of the iterable, if known
        mean_only : bool (Default = True)
            whether to generate the predicted variance in addition to the mean

        Returns
        -------
        means : np.ndarray
            the mean predicted values
        variances: np.ndarray
            the variance in the predicted means, empty if mean_only is True
        """
        if self.type_ == 'mpn':
            # MPNs predict directly on the input identifier
            xs = x_ids
            batched_size = None
        else:
            xs = x_feats

        if batched_size:
            n_batches = (size//batched_size) + 1 if size else None
        else:
            xs = batches(xs, self.test_batch_size)
            n_batches = (size//self.test_batch_size) + 1 if size else None
            batched_size = self.test_batch_size

        meanss = []
        variancess = []

        if mean_only:
            for batch_xs in tqdm(
                xs, total=n_batches, desc='Inference', smoothing=0.,
                unit='smi', unit_scale=batched_size, disable=disable
            ):
                means = self.get_means(batch_xs)
                meanss.append(means)
                variancess.append([])
        else:
            for batch_xs in tqdm(
                xs, total=n_batches, desc='Inference', smoothing=0.,
                unit='smi', unit_scale=batched_size, disable=disable
            ):
                means, variances = self.get_means_and_vars(batch_xs)
                meanss.append(means)
                variancess.append(variances)

        return np.concatenate(meanss), np.concatenate(variancess)
    
    @abstractmethod
    def save(self, path) -> str:
        """Save the model under path"""
    
    @abstractmethod
    def load(self, path):
        """load the model from path"""

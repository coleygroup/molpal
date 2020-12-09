"""This module contains Model implementations that utilize an NN model as their 
underlying model"""

from functools import partial
import logging
import os
from typing import (Callable, Iterable, List, NoReturn,
                    Optional, Sequence, Tuple, TypeVar)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import numpy as np
from numpy import ndarray
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from molpal.models.base import Model
from molpal.models.utils import feature_matrix

T = TypeVar('T')
T_feat = TypeVar('T_feat')
Dataset = tf.data.Dataset

class NN:
    """A feed-forward neural network model

    Attributes
    ----------
    model : keras.Sequential
        the underlying model on which to train and perform inference with
    optimizer : keras.optimizers.Adam
        the model optimizer
    loss : Callable
        the loss function to use
    input_size : int
        the dimension of the model inputs
    output_dim : int
        the dimension of the model outputs
    batch_size : int
        the size to batch training into
    dropout : Optional[float]
        If specified, add a dropout hidden layer with the specified dropout
        rate after each hidden layer
    dropout_at_predict : bool (Default = False)
       Whether to perform stochastic dropout during prediction
    mean : float
        the mean of the unnormalized data
    std : float
        the standard deviation of the unnormalized data
    ncpu : int
        the number of cores to parallelize feature matrix calculation over

    Parameters
    ----------
    input_size : int
    output_size : int
    batch_size : int (Default = 4096)
    layer_sizes : Optional[Sequence[int]] (Default = None)
        the sizes of the hidden layers in the network. If None, default to
        two hidden layers with 100 neurons each.
    dropout : Optional[float] (Default = None)
    dropout_at_predict : bool (Default = False)
    activation : Optional[str] (Default = 'relu')
        the name of the activation function to use
    ncpu : int (Default = 1)
        the number of cores to parallelize feature matrix calculation over
    """
    def __init__(self, input_size: int, output_size: int,
                 batch_size: int = 4096,
                 layer_sizes: Optional[Sequence[int]] = None,
                 dropout: Optional[float] = None,
                 dropout_at_predict: bool = False,
                 activation: Optional[str] = 'relu',
                 ncpu: int = 1):
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

        layer_sizes = layer_sizes or [100, 100]
        self.model, self.optimizer, self.loss = self.build(
            layer_sizes, dropout, dropout_at_predict, activation
        )

        self.mean = 0
        self.std = 0
        self.ncpu = ncpu

    def build(self, layer_sizes, dropout, dropout_at_predict, activation):
        """Build the model, optimizer, and loss function"""
        self.model = keras.Sequential()

        inputs = keras.layers.Input(shape=(self.input_size,))

        hidden = inputs
        for layer_size in layer_sizes:
            hidden = keras.layers.Dense(
                units=layer_size,
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(0.01),
            )(hidden)

            if dropout:
                hidden = keras.layers.Dropout(
                    dropout
                )(hidden, training=dropout_at_predict)

        outputs = keras.layers.Dense(
            self.output_size, activation='linear'
        )(hidden)

        model = keras.Model(inputs, outputs)

        if self.output_size == 1:
            optimizer = keras.optimizers.Adam(lr=0.01)
            loss = keras.losses.mse
        elif self.output_size == 2:
            # second output means we're using MVE approach
            optimizer = keras.optimizers.Adam(lr=0.05)
            def loss(y_true, y_pred):
                mu = y_pred[:, 0]
                var = tf.math.softplus(y_pred[:, 1])

                return tf.reduce_mean(
                    tf.math.log(2*3.141592)/2
                    + tf.math.log(var)/2
                    + tf.math.square(mu-y_true)/(2*var)
                )
        else:
            raise ValueError(
                f'NN output size ({self.output_size}) must be 1 or 2')

        return model, optimizer, loss

    def train(self, xs: Iterable[T], ys: Iterable[float],
              featurize: Callable[[T], ndarray]) -> bool:
        """Train the model on xs and ys with the given featurizer

        Parameters
        ----------
        xs : Sequence[T]
            an sequence of inputs in their identifier representations
        ys : Sequence[float]
            a parallel sequence of target values for these inputs
        featurize : Callable[[T], ndarray]
            a function that transforms an identifier into its uncompressed
            feature representation
        
        Returns
        -------
        True
        """
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        X = feature_matrix(xs, featurize, self.ncpu)
        Y = self._normalize(ys)

        self.model.fit(
            X, Y,
            batch_size=self.batch_size, validation_split=0.2, epochs=50,
            validation_freq=2, verbose=2,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5,
                    verbose=2, restore_best_weights=True),
                tfa.callbacks.TQDMProgressBar(leave_epoch_progress=False)
            ]
        )

        return True

    def predict(self, xs: Sequence[ndarray]) -> ndarray:
        X = np.stack(xs, axis=0)
        Y_pred = self.model.predict(X)

        if self.output_size == 1:
            Y_pred = Y_pred * self.std + self.mean
        else:
            Y_pred[:, 0] = Y_pred[:, 0] * self.std + self.mean
            Y_pred[:, 1] = Y_pred[:, 1] * self.std**2

        return Y_pred
    
    def save(self, path) -> None:
        self.model.save(path)
    
    def load(self, path) -> None:
        self.model = keras.models.load_model(path)
    
    def _normalize(self, ys: Sequence[float]) -> ndarray:
        Y = np.stack(list(ys))
        self.mean = np.nanmean(ys)
        self.std = np.nanstd(ys)

        return (Y - self.mean) / self.std

class NNModel(Model):
    """A simple feed-forward neural network model

    Attributes
    ----------
    model : Type[NN]
        the underlying neural net on which to train and perform inference
    
    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training
    ncpu : int (Default = 1)
        the number of cores to parallelize feature matrix calculation over
    
    See also
    --------
    NNDropoutModel
    NNEnsembleModel
    NNTwoOutputModel
    """
    def __init__(self, input_size: int, test_batch_size: Optional[int] = 4096,
                 dropout: Optional[float] = 0.0, ncpu: int = 1, **kwargs):
        test_batch_size = test_batch_size or 4096

        self.build_model = partial(NN, input_size=input_size, output_size=1,
                                   batch_size=test_batch_size, dropout=dropout,
                                   ncpu=ncpu)
        self.model = self.build_model()

        super().__init__(test_batch_size, **kwargs)

    @property
    def provides(self):
        return {'means'}

    @property
    def type_(self):
        return 'nn'

    def train(self, xs: Iterable[T], ys: Sequence[Optional[float]], *,
              featurize: Callable[[T], ndarray], retrain: bool = False) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys, featurize)

    def get_means(self, xs: List) -> ndarray:
        return self.model.predict(xs)[:, 0]

    def get_means_and_vars(self, xs: List) -> NoReturn:
        raise TypeError('NNModel can\'t predict variances!')

class NNEnsembleModel(Model):
    """A feed-forward neural network ensemble model for estimating mean
    and variance.
    
    Attributes
    ----------
    models : List[Type[NN]]
        the underlying neural nets on which to train and perform inference
    
    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training
    ensemble_size : int (Default = 5)
        the number of separate models to train
    ncpu : int (Default = 1)
        the number of cores to parallelize feature matrix calculation over
    bootstrap_ensemble : bool
        NOTE: UNUSED
    """
    def __init__(self, input_size: int, test_batch_size: Optional[int] = 4096,
                 dropout: Optional[float] = 0.0, ensemble_size: int = 5,
                 bootstrap_ensemble: Optional[bool] = False,
                 ncpu: int = 1, **kwargs):
        test_batch_size = test_batch_size or 4096
        self.build_model = partial(NN, input_size=input_size, output_size=1,
                                   batch_size=test_batch_size, dropout=dropout,
                                   ncpu=ncpu)

        self.ensemble_size = ensemble_size
        self.models = [self.build_model() for _ in range(self.ensemble_size)]

        self.bootstrap_ensemble = bootstrap_ensemble # TODO: Actually use this

        super().__init__(test_batch_size, **kwargs)

    @property
    def type_(self):
        return 'nn'

    @property
    def provides(self):
        return {'means', 'vars'}

    def train(self, xs: Iterable[T], ys: Sequence[Optional[float]], *,
              featurize: Callable[[T], ndarray], retrain: bool = False):
        if retrain:
            self.models = [
                self.build_model() for _ in range(self.ensemble_size)
            ]

        return all([model.train(xs, ys, featurize) for model in self.models])

    def get_means(self, xs: Sequence) -> np.ndarray:
        preds = np.zeros((len(xs), len(self.models)))
        for j, model in tqdm(enumerate(self.models), leave=False,
                             desc='ensemble prediction', unit='model'):
            preds[:, j] = model.predict(xs)[:, 0]

        return np.mean(preds, axis=1)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.zeros((len(xs), len(self.models)))
        for j, model in tqdm(enumerate(self.models), leave=False,
                             desc='ensemble prediction', unit='model'):
            preds[:, j] = model.predict(xs)[:, 0]

        return np.mean(preds, axis=1), np.var(preds, axis=1)

class NNTwoOutputModel(Model):
    """Feed forward neural network with two outputs so it learns to predict
    its own uncertainty at the same time
    
    Attributes
    ----------
    model : Type[NN]
        the underlying neural net on which to train and perform inference
    
    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training
    ncpu : int (Default = 0)
        the number of workers over which to parallelize
        feature matrix calculation
    """
    def __init__(self, input_size: int, test_batch_size: Optional[int] = 4096,
                 dropout: Optional[float] = 0.0, ncpu: int = 0, **kwargs):
        test_batch_size = test_batch_size or 4096

        self.build_model = partial(NN, input_size=input_size, output_size=2,
                                   batch_size=test_batch_size, dropout=dropout,
                                   ncpu=ncpu)
        self.model = self.build_model()

        super().__init__(test_batch_size, **kwargs)

    @property
    def type_(self):
        return 'nn'

    @property
    def provides(self):
        return {'means', 'vars'}

    def train(self, xs: Iterable[T], ys: Sequence[Optional[float]], *,
              featurize: Callable[[T], ndarray], retrain: bool = False) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys, featurize)

    def get_means(self, xs: Sequence) -> np.ndarray:
        preds = self.model.predict(xs)
        return preds[:, 0]

    def get_means_and_vars(self, xs: Sequence) -> Tuple[ndarray, ndarray]:
        preds = self.model.predict(xs)
        return preds[:, 0], self._safe_softplus(preds[:, 1])

    @classmethod
    def _safe_softplus(cls, xs):
        in_range = xs < 100
        return np.log(1 + np.exp(xs*in_range))*in_range + xs*(1 - in_range)

class NNDropoutModel(Model):
    """Feed forward neural network that uses MC dropout for UQ
    
    Attributes
    ----------
    model : Type[NN]
        the underlying neural net on which to train and perform inference
    dropout_size : int

    Parameters
    ----------
    input_size : int
        the size of the input dimension of the NN
    test_batch_size : Optional[int] (Defulat = 4096)
        the size into which inputs should be batched
        during training and inference
    dropout : Optional[float] (Default = 0.0)
        the dropout probability during training
    ncpu : int (Default = 1)
        the number of cores to parallelize feature matrix calculation over
    dropout_size : int (Default = 10)
        the number of passes to make through the network during inference
    """
    def __init__(self, input_size: int, test_batch_size: Optional[int] = 4096,
                 dropout: Optional[float] = 0.2, dropout_size: int = 10,
                 ncpu: int = 1, **kwargs):
        test_batch_size = test_batch_size or 4096

        self.build_model = partial(NN, input_size=input_size, output_size=1,
                                   batch_size=test_batch_size, dropout=dropout,
                                   dropout_at_predict=True, ncpu=ncpu)
        self.model = self.build_model()
        self.dropout_size = dropout_size

        super().__init__(test_batch_size, **kwargs)

    @property
    def type_(self):
        return 'nn'

    @property
    def provides(self):
        return {'means', 'vars', 'stochastic'}

    def train(self, xs: Iterable[T], ys: Sequence[Optional[float]], *,
              featurize: Callable[[T], ndarray], retrain: bool = False) -> bool:
        if retrain:
            self.model = self.build_model()
        
        return self.model.train(xs, ys, featurize)

    def get_means(self, xs: Sequence) -> ndarray:
        predss = self._get_predss(xs)
        return np.mean(predss, axis=1)

    def get_means_and_vars(self, xs: Sequence) -> Tuple[ndarray, ndarray]:
        predss = self._get_predss(xs)
        return np.mean(predss, axis=1), np.var(predss, axis=1)

    def _get_predss(self, xs: Sequence) -> ndarray:
        """Get the predictions for each dropout pass"""
        predss = np.zeros((len(xs), self.dropout_size))
        for j in tqdm(range(self.dropout_size), leave=False,
                      desc='bootstrap prediction', unit='pass'):
            predss[:, j] = self.model.predict(xs)[:, 0]

        return predss

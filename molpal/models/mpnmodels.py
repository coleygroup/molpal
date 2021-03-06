"""This module contains Model implementations that utilize the MPNN model as 
their underlying model"""
from functools import partial
import json
import os
from typing import Iterable, List, NoReturn, Optional, Sequence, Tuple, TypeVar

import numpy as np
from numpy import ndarray
import ray
from ray.util.sgd import TorchTrainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from tqdm import tqdm, trange

from .chemprop.data.data import (MoleculeDatapoint, MoleculeDataset,
                                 MoleculeDataLoader)
from .chemprop.data.utils import split_data
from . import chemprop

from molpal.models.base import Model
from molpal.models import mpnn, utils

T = TypeVar('T')
T_feat = TypeVar('T_feat')

class MPNN:
    """A message-passing neural network base class

    This class serves as a wrapper for the Chemprop MoleculeModel, providing
    convenience and modularity in addition to uncertainty quantification
    methods as originally implemented in the Chemprop confidence branch

    Attributes
    ----------
    model : MoleculeModel
        the underlying chemprop model on which to train and make predictions
    train_args : Namespace
        the arguments used for model training
    loss_func : Callable
        the loss function used in model training
    metric_func : str
        the metric function used in model evaluation
    device : str {'cpu', 'cuda'}
        the device on which training/evaluation/prediction is performed
    batch_size : int
        the size of each batch during training to update gradients
    epochs : int
        the number of epochs over which to train
    ncpu : int
        the number of cores over which to parallelize input batch preparation
    ddp : bool
        whether to train the model over a distributed setup. Only works with
        CUDA >= 11.0
    precision : int
        the precision with which to train the model represented in the number 
        of bits
    """
    def __init__(self, batch_size: int = 50,
                 uncertainty_method: Optional[str] = None,
                 dataset_type: str = 'regression', num_tasks: int = 1,
                 atom_messages: bool = False, hidden_size: int = 300,
                 bias: bool = False, depth: int = 3, dropout: float = 0.0,
                 undirected: bool = False, activation: str = 'ReLU',
                 ffn_hidden_size: Optional[int] = None,
                 ffn_num_layers: int = 2, metric: str = 'rmse',
                 epochs: int = 50, warmup_epochs: float = 2.0,
                 init_lr: float = 1e-4, max_lr: float = 1e-3,
                 final_lr: float = 1e-4, ncpu: int = 1,
                 ddp: bool = False, precision: int = 32):
        self.ncpu = ncpu
        self.ddp = ddp
        if precision not in (16, 32):
            raise ValueError(
                f'arg: "precision" can only be (16, 32). got: {precision}'
            )
        self.precision = precision

        self.model = mpnn.MoleculeModel(
            uncertainty_method=uncertainty_method,
            dataset_type=dataset_type, num_tasks=num_tasks,
            atom_messages=atom_messages, hidden_size=hidden_size,
            bias=bias, depth=depth, dropout=dropout, undirected=undirected,
            activation=activation, ffn_hidden_size=ffn_hidden_size, 
            ffn_num_layers=ffn_num_layers
        )

        self.uncertainty_method = uncertainty_method
        self.uncertainty = uncertainty_method in {'mve'}
        self.dataset_type = dataset_type
        self.num_tasks = num_tasks

        self.epochs = epochs
        self.batch_size = batch_size

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.metric = metric

        self.loss_func = mpnn.utils.get_loss_func(
            dataset_type, uncertainty_method)
        self.metric_func = chemprop.utils.get_metric_func(metric)
        self.scaler = None

        self.use_gpu = ray.cluster_resources().get('GPU', 0) > 0
        if self.use_gpu:
            _predict = ray.remote(num_cpus=ncpu, num_gpus=1)(mpnn.predict)
        else:
            _predict = ray.remote(num_cpus=ncpu)(mpnn.predict)
        
        self._predict = _predict

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, device):
        self.__device = device

    def train(self, smis: Iterable[str], targets: Sequence[float]) -> bool:
        """Train the model on the inputs SMILES with the given targets"""
        train_data, val_data = self.make_datasets(smis, targets)
        config = {
            'model': self.model,
            'dataset_type': self.dataset_type,
            'uncertainty_method': self.uncertainty_method,
            'warmup_epochs': self.warmup_epochs,
            'max_epochs': self.epochs,
            'init_lr': self.init_lr,
            'max_lr': self.max_lr,
            'final_lr': self.final_lr,
            'metric': self.metric,
        }
        if self.ddp:
            ngpu = int(ray.cluster_resources().get('GPU', 0))
            if ngpu > 0:
                num_workers = ngpu
            else:
                num_workers = ray.cluster_resources()['CPU'] // self.ncpu
            
            train_data, val_data = self.make_datasets(smis, targets)
            config['steps_per_epoch'] = (
                len(train_data) // (self.batch_size * num_workers)
            )
            config['train_loader'] = MoleculeDataLoader(
                dataset=train_data, batch_size=self.batch_size * num_workers,
                num_workers=self.ncpu, pin_memory=self.use_gpu
            )
            config['val_loader'] = MoleculeDataLoader(
                dataset=val_data, batch_size=self.batch_size * num_workers,
                num_workers=self.ncpu, pin_memory=self.use_gpu
            )

            trainer = TorchTrainer(
                training_operator_cls=mpnn.MPNNOperator,
                num_workers=num_workers, config=config,
                use_gpu=self.use_gpu, scheduler_step_freq='batch'
            )
            
            pbar = trange(self.epochs, desc='Training', unit='epoch')
            for i in pbar:
                train_loss = trainer.train()['train_loss']
                val_res = trainer.validate()
                val_loss = val_res['val_loss']
                lr = val_res['lr']
                pbar.set_postfix_str(
                    f'loss={train_loss:0.3f}, '
                    f'val_loss={val_loss:0.3f} '
                    f'lr={lr}'
                )
                print(f'Epoch {i}: lr={lr}', flush=True)

            self.model = trainer.get_model()
            return True

        train_dataloader = MoleculeDataLoader(
            dataset=train_data, batch_size=self.batch_size,
            num_workers=self.ncpu, pin_memory=self.use_gpu
        )
        val_dataloader = MoleculeDataLoader(
            dataset=val_data, batch_size=self.batch_size,
            num_workers=self.ncpu, pin_memory=self.use_gpu
        )
        model = mpnn.LitMPNN(config)
        early_stop_callback = EarlyStopping(
            monitor='val_loss', patience=10, verbose=True, mode='min'
        )
        trainer = pl.Trainer(
            max_epochs=self.epochs, callbacks=[early_stop_callback],
            gpus=1 if self.use_gpu else 0, precision=self.precision,
            progress_bar_refresh_rate=100
        )
        trainer.fit(model, train_dataloader, val_dataloader)
        
        return True

    def make_datasets(
            self, xs: Iterable[str], ys: Sequence[float]
        ) -> Tuple[MoleculeDataset, MoleculeDataset]:
        """Split xs and ys into train and validation datasets"""

        data = MoleculeDataset([
            MoleculeDatapoint(smiles=[x], targets=[y])
            for x, y in zip(xs, ys)
        ])
        train_data, val_data, _ = split_data(data=data, sizes=(0.8, 0.2, 0.0))

        self.scaler = train_data.normalize_targets()    
        val_data.scale_targets(self.scaler)

        return train_data, val_data

    def predict(self, smis: Iterable[str]) -> ndarray:
        """Generate predictions for the inputs xs"""
        smis_batches = utils.batches(smis, 20000)

        model = ray.put(self.model)
        scaler = ray.put(self.scaler)
        refs = [
            self._predict.remote(
                model, smis, self.batch_size, self.ncpu,
                self.uncertainty, scaler, self.use_gpu, True
            ) for smis in smis_batches
        ]
        preds_chunks = [
            ray.get(r) for r in tqdm(refs, desc='Prediction', leave=False)
        ]
        preds_chunks = np.column_stack(preds_chunks)

        return preds_chunks

        # test_data = MoleculeDataset([MoleculeDatapoint(smiles=[smi]) for smi in smis])
        # data_loader = MoleculeDataLoader(
        #     dataset=test_data,
        #     batch_size=self.batch_size,
        #     num_workers=self.ncpu
        # )

        # return mpnn.predict(self.model, data_loader, scaler=self.scaler)

    def save(self, path) -> str:
        model_path = f'{path}/model.pt'
        torch.save(self.model.state_dict(), model_path)

        state_path = f'{path}/state.json'
        state = {
            'stds': self.scaler.stds.tolist(),
            'means': self.scaler.means.tolist(),
            'model_path': model_path
        }
        json.dump(state, open(state_path, 'w'))

        return state_path
    
    def load(self, path):
        self.model.load(path)

class MPNModel(Model):
    """Message-passing model that learns feature representations of inputs and
    passes these inputs to a feed-forward neural network to predict means"""
    def __init__(self, test_batch_size: Optional[int] = 1000000,
                 ncpu: int = 1, ddp: bool = False, precision: int = 32, 
                 **kwargs):
        test_batch_size = test_batch_size or 1000000

        self.build_model = partial(
            MPNN, ncpu=ncpu, ddp=ddp, precision=precision
        )
        self.model = self.build_model()

        super().__init__(test_batch_size=test_batch_size, **kwargs)

    @property
    def provides(self):
        return {'means'}

    @property
    def type_(self):
        return 'mpn'

    def train(self, xs: Iterable[str], ys: Sequence[float], *,
              retrain: bool = False, **kwargs) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys)

    def get_means(self, xs: Sequence[str]) -> ndarray:
        preds = self.model.predict(xs)
        return preds

    def get_means_and_vars(self, xs: List) -> NoReturn:
        raise TypeError('MPNModel cannot predict variance!')

    def save(self, path) -> str:
        return self.model.save(path)
    
    def load(self, path):
        self.model.load(path)

class MPNDropoutModel(Model):
    """Message-passing network model that predicts means and variances through
    stochastic dropout during model inference"""
    def __init__(self, test_batch_size: Optional[int] = 1000000,
                 dropout: float = 0.2, dropout_size: int = 10,
                 ncpu: int = 1, ddp: bool = False, precision: int = 32,
                 **kwargs):
        test_batch_size = test_batch_size or 1000000

        self.build_model = partial(
            MPNN, uncertainty_method='dropout', dropout=dropout, 
            ncpu=ncpu, ddp=ddp, precision=precision
        )
        self.model = self.build_model()

        self.dropout_size = dropout_size

        super().__init__(test_batch_size=test_batch_size, **kwargs)
    
    @property
    def type_(self):
        return 'mpn'

    @property
    def provides(self):
        return {'means', 'vars', 'stochastic'}

    def train(self, xs: Iterable[str], ys: Sequence[float], *,
              retrain: bool = False, **kwargs) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys)

    def get_means(self, xs: Sequence[str]) -> ndarray:
        predss = self._get_predictions(xs)
        return np.mean(predss, axis=1)

    def get_means_and_vars(self, xs: Sequence[str]) -> Tuple[ndarray, ndarray]:
        predss = self._get_predictions(xs)
        return np.mean(predss, axis=1), np.var(predss, axis=1)

    def _get_predictions(self, xs: Sequence[str]) -> ndarray:
        predss = np.zeros((len(xs), self.dropout_size))
        for j in tqdm(range(self.dropout_size),
                      desc='dropout prediction'):
            predss[:, j] = self.model.predict(xs)
        return predss

    def save(self, path) -> str:
        return self.model.save(path)
    
    def load(self, path):
        self.model.load(path)

class MPNTwoOutputModel(Model):
    """Message-passing network model that predicts means and variances
    through mean-variance estimation"""
    def __init__(self, test_batch_size: Optional[int] = 1000000,
                 ncpu: int = 1, ddp: bool = False, precision: int = 32,
                 **kwargs):
        test_batch_size = test_batch_size or 1000000

        self.build_model = partial(
            MPNN, uncertainty_method='mve', ncpu=ncpu,
            ddp=ddp, precision=precision
        )
        self.model = self.build_model()

        super().__init__(test_batch_size=test_batch_size, **kwargs)

    @property
    def type_(self):
        return 'mpn'

    @property
    def provides(self):
        return {'means', 'vars'}

    def train(self, xs: Iterable[str], ys: Sequence[float], *,
              retrain: bool = False, **kwargs) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys)

    def get_means(self, xs: Sequence[str]) -> ndarray:
        means, _ = self._get_predictions(xs)
        return means.flatten()

    def get_means_and_vars(self, xs: Sequence[str]) -> Tuple[ndarray, ndarray]:
        means, variances = self._get_predictions(xs)
        return means.flatten(), variances.flatten()

    def _get_predictions(self, xs: Sequence[str]) -> Tuple[ndarray, ndarray]:
        """Get both the means and the variances for the xs"""
        means, variances = self.model.predict(xs)
        return means, variances

    def save(self, path) -> str:
        return self.model.save(path)
    
    def load(self, path):
        self.model.load(path)
# def combine_sds(sd1: float, mu1: float, n1: int,
#                 sd2: float, mu2: float, n2: int):

#     var1 = sd1**2
#     var2 = sd2**2
#     n_total = n1 + n2
#     mu_combined = (n1*mu1 + n2*mu2) / n_total

#     sd_combined = sqrt(
#         (n1*(var1 + (mu1-mu_combined)**2) + n2*(var2 + (mu2-mu_combined)**2))
#         / n_total
#     )
#     return sd_combined

"""This module contains Model implementations that utilize the MPNN model as 
their underlying model"""

from argparse import Namespace
from functools import partial
from itertools import chain
from typing import Iterable, List, NoReturn, Optional, Sequence, Tuple, TypeVar

import numpy as np
from numpy import ndarray
import ray
from ray.util.sgd import TorchTrainer
from tqdm import tqdm, trange
import torch.cuda

from .chemprop.data.data import (MoleculeDatapoint, MoleculeDataset,
                                 MoleculeDataLoader)
from .chemprop.data.scaler import StandardScaler
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
                 final_lr: float = 1e-4, log_frequency: int = 10,
                 ncpu: int = 1):
        if torch.cuda.is_available():
            device = 'cuda' 
        else:
            device = 'cpu'

        self.ncpu = ncpu

        self.model = mpnn.MoleculeModel(
            uncertainty_method=uncertainty_method,
            dataset_type=dataset_type, num_tasks=num_tasks,
            atom_messages=atom_messages, hidden_size=hidden_size,
            bias=bias, depth=depth, dropout=dropout, undirected=undirected,
            activation=activation, ffn_hidden_size=ffn_hidden_size, 
            ffn_num_layers=ffn_num_layers, device=device,
        )
        self.uncertainty = uncertainty_method in {'mve'}
        self.device = device

        self.epochs = epochs
        self.batch_size = batch_size
        self.train_args = Namespace(
            dataset_type='regression', metric=metric,
            epochs=epochs, warmup_epochs=warmup_epochs,
            total_epochs=epochs, batch_size=batch_size,
            init_lr=init_lr, max_lr=max_lr, final_lr=final_lr, num_lrs=1,
            log_frequency=log_frequency
        )

        self.loss_func = mpnn.utils.get_loss_func(
            dataset_type, uncertainty_method)
        self.metric_func = chemprop.utils.get_metric_func(metric)
        self.scaler = None

        if ray.cluster_resources().get('GPU', 0) > 0:
            @ray.remote(num_cpus=ncpu, num_gpus=1)
            def _predict(model, smis, batch_size, ncpu, scaler):
                model.device = 0

                test_data = MoleculeDataset(
                    [MoleculeDatapoint(smiles=[smi]) for smi in smis]
                )
                data_loader = MoleculeDataLoader(
                    dataset=test_data, batch_size=batch_size, num_workers=ncpu
                )
                return mpnn.predict(
                    model, data_loader, self.uncertainty,
                    disable=True, scaler=scaler
                )
        else:
            @ray.remote(num_cpus=ncpu)
            def _predict(model, smis, batch_size, ncpu, scaler):
                model.device = 'cpu'

                test_data = MoleculeDataset(
                    [MoleculeDatapoint(smiles=[smi]) for smi in smis]
                )
                data_loader = MoleculeDataLoader(
                    dataset=test_data, batch_size=batch_size, num_workers=ncpu
                )
                return mpnn.predict(
                    model, data_loader, self.uncertainty,
                    disable=True, scaler=scaler
                )
        
        self._predict = _predict

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, device):
        self.__device = device
        self.model.device = device

    def train(self, smis: Iterable[str], targets: Sequence[float]) -> bool:
        """Train the model on the inputs SMILES with the given targets"""
        config = {
            'smis': smis, 'targets': targets, 'model': self.model,
            'uncertainty': self.uncertainty, 'train_args': self.train_args, 
            'ncpu': self.ncpu, 
        }
        ngpu = int(ray.cluster_resources().get('GPU', 0))
        if ngpu > 0:
            num_workers = ngpu #- 1
        else:
            num_workers = ray.cluster_resources()['CPU'] // self.ncpu

        trainer = TorchTrainer(
            training_operator_cls=mpnn.MPNNOperator, config=config,
            num_workers=num_workers, use_gpu=ngpu>0, #backend='gloo'
            scheduler_step_freq='manual'
        )

        
        for _ in trange(self.epochs, desc='Training', unit='epoch'):
            trainer.train()
            trainer.validate()
        self.model = trainer.get_model()
        #print(metricss)
        #print(val_metricss)
        return True

        train_data, val_data = self.make_datasets(smis, targets)
        n_xs = len(train_data) + len(val_data)
        self.train_args.train_data_size = n_xs

        train_data_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=self.batch_size,
            num_workers=self.ncpu
        )
        val_data_loader = MoleculeDataLoader(
            dataset=val_data,
            batch_size=self.batch_size,
            num_workers=self.ncpu
        )

        optimizer = chemprop.utils.build_optimizer(
            self.model, self.train_args)
        scheduler = chemprop.utils.build_lr_scheduler(
            optimizer, self.train_args)

        n_iter = 0
        best_val_score = float('-inf')
        best_state_dict = self.model.state_dict()

        for _ in trange(self.epochs, desc='Training', unit='epoch'):
            n_iter = mpnn.train(
                self.model, train_data_loader, self.loss_func,
                optimizer, scheduler, self.train_args, n_iter
            )
            # if isinstance(scheduler, exponentialLR)
            #   scheduler.step()
            # this will need to be changed if we allow classification
            val_scores = mpnn.evaluate(
                self.model, val_data_loader, 1,
                self.metric_func, 'regession', self.scaler)
            val_score = np.nanmean(val_scores)

            if val_score > best_val_score:
                best_val_score = val_score
                best_state_dict = self.model.state_dict()

        self.model.load_state_dict(best_state_dict)

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

        train_targets = train_data.targets()
        self.scaler = StandardScaler().fit(train_targets)

        scaled_targets = self.scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

        return train_data, val_data

    def predict(self, smis: Iterable[str]) -> ndarray:
        """Generate predictions for the inputs xs"""
        smis_batches = utils.batches(smis, 10000)

        model = ray.put(self.model)
        scaler = ray.put(self.scaler)
        refs = [
            self._predict.remote(
                model, smis, self.batch_size, self.ncpu, scaler
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

class MPNModel(Model):
    """Message-passing model that learns feature representations of inputs and
    passes these inputs to a feed-forward neural network to predict means"""
    def __init__(self, test_batch_size: Optional[int] = 1000000,
                 ncpu: int = 1, **kwargs):
        test_batch_size = test_batch_size or 1000000

        self.build_model = partial(MPNN, ncpu=ncpu)
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

class MPNDropoutModel(Model):
    """Message-passing network model that predicts means and variances through
    stochastic dropout during model inference"""
    def __init__(self, test_batch_size: Optional[int] = 1000000,
                 dropout: float = 0.2, dropout_size: int = 10,
                 ncpu: int = 1, **kwargs):
        test_batch_size = test_batch_size or 1000000

        self.build_model = partial(MPNN, uncertainty_method='dropout',
                                   dropout=dropout, ncpu=ncpu)
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

class MPNTwoOutputModel(Model):
    """Message-passing network model that predicts means and variances
    through mean-variance estimation"""
    def __init__(self, test_batch_size: Optional[int] = 1000000,
                 ncpu: int = 1, **kwargs):
        test_batch_size = test_batch_size or 1000000

        self.build_model = partial(MPNN, uncertainty_method='mve', ncpu=ncpu)
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

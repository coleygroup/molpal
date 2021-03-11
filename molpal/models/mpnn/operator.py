from argparse import Namespace
from typing import Iterable, Optional, Sequence, Tuple, TypeVar

import numpy as np
from ray.util.sgd.torch import TrainingOperator
import torch
import torch.cuda

from ..chemprop.data.data import (MoleculeDatapoint, MoleculeDataset,
                                 MoleculeDataLoader)
from ..chemprop.data.scaler import StandardScaler
from ..chemprop.data.utils import split_data
from .. import chemprop

from molpal.models import mpnn

T = TypeVar('T')
T_feat = TypeVar('T_feat')

class MPNNOperator(TrainingOperator):
    def setup(self, config):
        smis = config['smis']
        targets = config['targets']

        self.scaler = None
        train_data, val_data = self.make_datasets(smis, targets)
        
        train_args = config['train_args']
        train_args.train_data_size = len(train_data) + len(val_data)
        self.train_args = train_args
        
        train_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=train_args.batch_size,
            num_workers=config.get('ncpu', 1)
        )
        val_loader = MoleculeDataLoader(
            dataset=val_data,
            batch_size=train_args.batch_size,
            num_workers=config.get('ncpu', 1)
        )

        model = config['model']
        optimizer = chemprop.utils.build_optimizer(model, train_args)
        scheduler = chemprop.utils.build_lr_scheduler(optimizer, train_args)
        criterion = mpnn.utils.get_loss_func(
            train_args.dataset_type, model.uncertainty_method
        )

        self.uncertainty = config['uncertainty']
        self.metric_func = chemprop.utils.get_metric_func(train_args.metric)

        self.n_iter = 0
        self.best_val_score = float('-inf')
        self.best_state_dict = model.state_dict()

        self.model, self.optimizer, self.criterion, self.scheduler = \
            self.register(
                models=model, optimizers=optimizer,
                criterion=criterion, schedulers=scheduler
            )
        self.register_data(
            train_loader=train_loader, validation_loader=val_loader
        )
    
    def train_epoch(self, train_loader, info):
        #train_loader = [x for x in train_loader]
        #print(train_loader)
        self.n_iter += mpnn.train(
            self.model, train_loader, self.criterion,
            self.optimizer, self.scheduler, self.uncertainty, self.n_iter, True
        )
        return {}
    
    def validate(self, val_loader, info):
        val_loader = [item for item in val_loader]
        #print(val_loader)
        val_scores = mpnn.evaluate(
            self.model, val_loader, 1, self.uncertainty,
            self.metric_func, 'regession', self.scaler)
        val_score = np.nanmean(val_scores)

        if val_score > self.best_val_score:
            self.best_val_score = val_score
            self.best_state_dict = self.model.state_dict()

        return {}
    def get_model(self):
        model = super().get_model()
        model.load_state_dict(self.best_state_dict)

        return model

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

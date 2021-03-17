from argparse import Namespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from ray.util.sgd.torch import TrainingOperator
import torch
from torch import nn, cuda

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
        train_data_size = len(train_data) + len(val_data)

        # scheduler_args = config['scheduler_args']
        # self.scheduler_args = scheduler_args
        
        train_loader = MoleculeDataLoader(
            dataset=train_data,
            batch_size=config['batch_size'],
            num_workers=config.get('ncpu', 1)
        )
        val_loader = MoleculeDataLoader(
            dataset=val_data,
            batch_size=config['batch_size'],
            num_workers=config.get('ncpu', 1)
        )

        model = config['model']
        optimizer = chemprop.utils.build_optimizer(
            model, config['init_lr']
        )
        scheduler = chemprop.utils.build_lr_scheduler(
            optimizer, train_data_size=train_data_size,
            **config['scheduler_args'])
        criterion = mpnn.utils.get_loss_func(
            config['dataset_type'], config['uncertainty_method']
        )

        self.uncertainty = config['uncertainty_method'] in {'mve'}
        self.metric_func = chemprop.utils.get_metric_func(config['metric'])

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
    
    def train_batch(self, batch: MoleculeDataset, batch_info: Dict) -> Dict:
        mol_batch = batch.batch_graph()
        
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion

        preds = model(mol_batch)
        targets = batch.targets()
        
        mask = torch.tensor(
            [list(map(bool, ys)) for ys in targets], device=self.device
        )
        targets = torch.tensor(
            [[y or 0 for y in ys] for ys in targets], device=self.device
        )
        class_weights = torch.ones(targets.shape, device=self.device)
        
        # if args.dataset_type == 'multiclass':
        #     targets = targets.long()
        #     loss = (torch.cat([
        #         loss_func(preds[:, target_index, :],
        #                    targets[:, target_index]).unsqueeze(1)
        #         for target_index in range(preds.size(1))
        #         ], dim=1) * class_weights * mask
        #     )

        if self.uncertainty:
            pred_means = preds[:, 0::2]
            pred_vars = preds[:, 1::2]

            loss = self.criterion(pred_means, pred_vars, targets)
        else:
            loss = self.criterion(preds, targets) * class_weights * mask

        loss = loss.sum() / mask.sum()
        return loss
    
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
    
    def validation_step(self, batch: MoleculeDataset, batch_idx) -> List[float]:
        preds = self.mpnn(batch.batch_graph())

        if self.uncertainty:
            preds = preds[:, 0::2]

        # preds_ = preds.cpu().numpy()
        targets_ = batch.targets()
        targets = torch.tensor(targets_, device=self.device)

        res = torch.sqrt(nn.MSELoss()(preds, targets))
        return res
    
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

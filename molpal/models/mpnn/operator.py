from typing import Dict, Iterable, List, Sequence, Tuple, TypeVar

from ray.util.sgd.torch import TrainingOperator
import torch
from torch.nn import functional as F
from torch.optim import Adam

from ..chemprop.data.data import MoleculeDatapoint, MoleculeDataset
from ..chemprop.data.scaler import StandardScaler
from ..chemprop.data.utils import split_data
from ..chemprop.nn_utils import NoamLR

from molpal.models import mpnn

T = TypeVar('T')
T_feat = TypeVar('T_feat')

class MPNNOperator(TrainingOperator):
    def setup(self, config):
        warmup_epochs = config.get('warmup_epochs', 2.)
        steps_per_epoch = config['steps_per_epoch']
        max_epochs = config['max_epochs']
        num_lrs = 1
        init_lr = config.get('init_lr', 1e-4)
        max_lr = config.get('max_lr', 1e-3)
        final_lr = config.get('final_lr', 1e-4)
        uncertainty_method = config.get('uncertainty_method', 'none')

        model = config['model']
        optimizer = Adam([{
            'params': model.parameters(), 'lr': init_lr, 'weight_decay': 0
        }])
        # optimizer = Adam(model.parameters(), init_lr, weight_decay=0)
        scheduler = NoamLR(
            optimizer=optimizer, warmup_epochs=[warmup_epochs],
            total_epochs=[max_epochs] * num_lrs,
            steps_per_epoch=steps_per_epoch,
            init_lr=[init_lr], max_lr=[max_lr], final_lr=[final_lr]
        )
        criterion = mpnn.utils.get_loss_func(
            config['dataset_type'], uncertainty_method
        )

        self.uncertainty = uncertainty_method in {'mve'}

        self.metric = {
            'mse': lambda X, Y: F.mse_loss(X, Y, reduction='none'),
            'rmse': lambda X, Y: torch.sqrt(F.mse_loss(X, Y, reduction='none'))
        }[config.get('metric', 'rmse')]

        train_loader = config['train_loader']
        val_loader = config['val_loader']

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
        
    def train_batch(self, batch: MoleculeDataset, batch_info: Dict) -> Dict:
        componentss, targets = batch

        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        scheduler = self.scheduler
        
        optimizer.zero_grad()

        # look @ "non_blocking=True" if weird things start happening
        device = 'cuda' if self.use_gpu else 'cpu'
        componentss = [[
            X.to(device, non_blocking=True)
                if isinstance(X, torch.Tensor) else X
            for X in components
        ] for components in componentss]

        mask = torch.tensor(
            [[bool(y) for y in ys] for ys in targets], device=device
        )
        targets = torch.tensor(
            [[y or 0 for y in ys] for ys in targets], device=device
        )
        class_weights = torch.ones(targets.shape, device=device)

        preds = model(componentss)
        # if self.dataset_type == 'multiclass':
        #     targets = targets.long()
        #     loss = (torch.cat([
        #         criterion(
        #             preds[:, target_index, :], targets[:, target_index]
        #         ).unsqueeze(1)
        #         for target_index in range(preds.size(1))
        #     ], dim=1) * class_weights * mask)

        if self.uncertainty:
            pred_means = preds[:, 0::2]
            pred_vars = preds[:, 1::2]

            loss = criterion(pred_means, pred_vars, targets)
        else:
            loss = criterion(preds, targets) * class_weights * mask

        loss = loss.sum() / mask.sum()

        loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()

        return {'train_loss': loss.item(), 'num_samples': len(targets)}
    
    def validate(self, val_iterator, info) -> Dict:
        """Runs one standard validation pass over the val_iterator.
        
        Parameters
        ----------
        val_iterator : Iterable
            Iterable constructed from the validation dataloader.
        info : Dict
            Dictionary for information to be used for custom validation 
            operations.

        Returns
        --------
        Dict
            A dict of metrics from the evaluation. By default, returns 
            "val_accuracy" and "val_loss" which is computed by aggregating 
            "loss" and "correct" values from ``validate_batch`` and dividing it 
            by the sum of ``num_samples`` from all calls to ``self.
            validate_batch``.
        """
        self.model.eval()

        losses = []
        num_samples = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iterator):
                batch_info = {"batch_idx": batch_idx}
                batch_info.update(info)
                step_results = self.validate_batch(batch, batch_info)
                losses.append(step_results['loss'])
                num_samples += step_results['num_samples']

            val_loss = torch.cat(losses).mean().item()

        return {
            'val_loss': val_loss, #/ num_samples,
            'num_samples': num_samples,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def validate_batch(self, batch: MoleculeDataset, batch_idx) -> List[float]:
        componentss, targets = batch

        model = self.model
        metric = self.metric

        device = 'cuda' if self.use_gpu else 'cpu'
        componentss = [[
            X.to(device, non_blocking=True)
                if isinstance(X, torch.Tensor) else X
            for X in components
        ] for components in componentss]
        targets = torch.tensor(targets, device=device)

        preds = model(componentss)
        if self.uncertainty:
            preds = preds[:, 0::2]

        loss = metric(preds, targets)
        return {'loss': loss, 'num_samples': len(targets)}
    
    # def get_model(self):
    #     model = super().get_model()
    #     model.load_state_dict(self.best_state_dict)

    #     return model

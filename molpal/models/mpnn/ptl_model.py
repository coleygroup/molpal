from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F

from molpal.models import mpnn
from ..chemprop.data import MoleculeDataset
from ..chemprop.nn_utils import NoamLR
from .. import chemprop

class LitMPNN(pl.LightningModule):
    """A message-passing neural network base class"""
    def __init__(self, config: Dict,
                #  model: nn.Module,
                #  uncertainty_method: Optional[str] = None,
                #  dataset_type: str = 'regression',
                #  warmup_epochs: float = 2.0,
                #  init_lr: float = 1e-4,
                #  max_lr: float = 1e-3,
                #  final_lr: float = 1e-4,
                #  metric: str = 'rmse'
                ):
        super().__init__()

        model = config['model']
        dataset_type = config['dataset_type']
        uncertainty_method = config.get('uncertainty_method', 'none')

        self.mpnn = model
        self.uncertainty = uncertainty_method in {'mve'}

        self.warmup_epochs = config.get('warmup_epochs', 2.)
        self.steps_per_epoch = config['steps_per_epoch']
        self.max_epochs = config['max_epochs']
        self.num_lrs = 1
        self.init_lr = config.get('init_lr', 1e-4)
        self.max_lr = config.get('max_lr', 1e-3)
        self.final_lr = config.get('final_lr', 1e-4)

        self.criterion = mpnn.utils.get_loss_func(
            dataset_type, uncertainty_method
        )
        # self.metric_func = chemprop.utils.get_metric_func(metric)
        self.metric = {
            'mse': lambda X, Y: F.mse_loss(X, Y, reduction='none'),
            'rmse': lambda X, Y: torch.sqrt(F.mse_loss(X, Y, reduction='none'))
        }[config.get('metric', 'rmse')]

    def training_step(self, batch: Tuple, batch_idx) -> torch.Tensor:
        componentss, targets = batch
        
        preds = self.mpnn(componentss)
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
        print(loss.item())
        return loss
    
    def validation_step(self, batch: Tuple, batch_idx) -> List[float]:
        componentss, targets = batch

        preds = self.mpnn(componentss)
        if self.uncertainty:
            preds = preds[:, 0::2]

        targets = torch.tensor(targets, device=self.device)

        return self.metric(preds, targets)

    def validation_epoch_end(self, outputs):
        val_loss = torch.cat(outputs).mean()
        # print(val_loss.item())
        self.log('val_loss', val_loss, prog_bar=True)

    def configure_optimizers(self) -> List:
        opt = Adam([{
            'params': self.mpnn.parameters(),
            'lr': self.init_lr,
            'weight_decay': 0
        }])
        sched = NoamLR(
            optimizer=opt,
            warmup_epochs=[self.warmup_epochs],
            total_epochs=[self.trainer.max_epochs] * self.num_lrs,
            steps_per_epoch=self.num_training_steps,
            init_lr=[self.init_lr],
            max_lr=[self.max_lr],
            final_lr=[self.final_lr]
        )
        scheduler = {
            'scheduler': sched,
            'interval': 'step' if isinstance(sched, NoamLR) else 'batch'
        }
        
        return [opt], [scheduler]

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())

        if isinstance(limit_batches, int):
            batches = min(batches, limit_batches)
        else:
            batches = int(limit_batches * batches)     

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

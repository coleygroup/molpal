
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam

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
        uncertainty_method = config['uncertainty_method']
        dataset_type = config['dataset_type']
        warmup_epochs = config['warmup_epochs']
        init_lr = config['init_lr']
        max_lr = config['max_lr']
        final_lr = config['final_lr']
        metric = config['metric']

        self.mpnn = model
        self.uncertainty = uncertainty_method in {'mve'}

        self.warmup_epochs = warmup_epochs
        self.steps_per_epoch = config['steps_per_epoch']
        self.max_epochs = config['max_epochs']
        self.num_lrs = 1
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

        self.criterion = mpnn.utils.get_loss_func(
            dataset_type, uncertainty_method
        )
        self.metric_func = chemprop.utils.get_metric_func(metric)

    def training_step(self, batch: MoleculeDataset, batch_idx) -> torch.Tensor:
        mol_batch = batch.batch_graph()
        
        preds = self.mpnn(mol_batch)
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
    
    def validation_step(self, batch: MoleculeDataset, batch_idx) -> List[float]:
        preds = self.mpnn(batch.batch_graph())

        if self.uncertainty:
            preds = preds[:, 0::2]

        # preds_ = preds.cpu().numpy()
        targets_ = batch.targets()
        targets = torch.tensor(targets_, device=self.device)

        res = torch.sqrt(nn.MSELoss()(preds, targets))
        return res

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack(outputs).cpu().numpy().mean()
        self.log('avg_val_loss', avg_val_loss)

    def configure_optimizers(self) -> List:
        opt = Adam([{
            'params': self.mpnn.parameters(),
            'lr': self.init_lr,
            'weight_decay': 0
        }])
        sched = NoamLR(
            optimizer=opt,
            warmup_epochs=[self.warmup_epochs],
            total_epochs=[self.max_epochs] * self.num_lrs,
            #total_epochs=[self.trainer.num_epochs] * self.num_lrs
            steps_per_epoch=self.steps_per_epoch,
            # steps_per_epoch=self.num_training_steps,
            init_lr=[self.init_lr],
            max_lr=[self.max_lr],
            final_lr=[self.final_lr]
        )
        return [opt], [sched]

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

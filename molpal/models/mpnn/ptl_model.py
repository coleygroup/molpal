
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import Adam

from molpal.models import mpnn
from ..chemprop.data import StandardScaler, MoleculeDataset, MoleculeDataLoader
from ..chemprop.nn_utils import NoamLR
from ..chemprop.data.data import (MoleculeDatapoint, MoleculeDataset,
                                 MoleculeDataLoader)
from ..chemprop.data.scaler import StandardScaler
from ..chemprop.data.utils import split_data
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

        # num_tasks = preds.shape[1]

        # # valid_preds and valid_targets have shape (num_tasks, data_size)
        # valid_preds = [[]] * num_tasks
        # valid_targets = [[]] * num_tasks
        # for j in range(num_tasks):
        #     for i in range(len(preds_)):
        #         valid_preds[j].append(preds_[i][j])
        #         valid_targets[j].append(targets[i][j])

        # # Compute metric
        # results = []
        # for preds, targets in zip(valid_preds, valid_targets):
        #     # if all targets or preds are identical classification will crash
        #     # if dataset_type == 'classification':
        #     #     if all(t == 0 for t in targets) or all(targets):
        #     #         # info('Warning: Found a task with targets all 0s or all 1s')
        #     #         results.append(float('nan'))
        #     #         continue
        #     #     if all(p == 0 for p in preds) or all(preds):
        #     #         # info('Warning: Found a task with predictions all 0s or all 1s')
        #     #         results.append(float('nan'))
        #     #         continue

        #     if len(targets) == 0:
        #         continue

        #     results.append(self.metric_func(targets, preds))

        # # print(f'val_loss: {results}')
        # self.log('val_loss', results)
        # print(results)
        # return results

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
            total_epochs=[self.max_epochs] * self.num_lrs, #self.trainer.*
            steps_per_epoch=self.steps_per_epoch, #self.num_training_steps,
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

    # def predict(self, data_loader: MoleculeDataLoader,
    #             scaler: Optional[StandardScaler] = None,
    #             disable: bool = False) -> List[List[float]]:
    #     self.mpnn.eval()

    #     pred_batches = []
    #     with torch.no_grad():
    #         for batch in tqdm(data_loader, desc='Inference', unit='batch',
    #                           leave=False, disable=disable):
    #             batch_graph = batch.batch_graph()
    #             pred_batch = self.mpnn(batch_graph)
    #             pred_batches.append(pred_batch.data.cpu().numpy())
    #     preds = np.concatenate(pred_batches)

    #     if self.uncertainty:
    #         means = preds[:, 0::2]
    #         variances = preds[:, 1::2]

    #         if scaler:
    #             means = scaler.inverse_transform(means)
    #             variances = scaler.stds**2 * variances

    #         return means, variances

    #     if scaler:
    #         preds = scaler.inverse_transform(preds)

    #     return preds

    # def make_datasets(
    #         self, xs: Iterable[str], ys: Sequence[float]
    #     ) -> Tuple[MoleculeDataset, MoleculeDataset]:
    #     """Split xs and ys into train and validation datasets"""

    #     data = MoleculeDataset([
    #         MoleculeDatapoint(smiles=[x], targets=[y])
    #         for x, y in zip(xs, ys)
    #     ])
    #     train_data, val_data, _ = split_data(data=data, sizes=(0.8, 0.2, 0.0))

    #     self.scaler = train_data.normalize_targets()    
    #     val_data.scale_targets(self.scaler)

    #     return train_data, val_data

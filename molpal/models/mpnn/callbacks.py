import sys
from typing import Optional

from pytorch_lightning.callbacks import ProgressBarBase
from tqdm import tqdm

class EpochAndStepProgressBar(ProgressBarBase):
    def __init__(self, refresh_rate: int = 100):
        super().__init__()
        self.step_bar = None
        self.epoch_bar = None
        self.refresh_rate = refresh_rate

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        self.epoch_bar = tqdm(
            desc='Training', unit='epoch', leave=True, dynamic_ncols=True,
            total=trainer.max_epochs,
        )
        self.step_bar = tqdm(
            desc='Epoch (train)', unit='step',
            leave=False, dynamic_ncols=True,
            total=self.total_train_batches+self.total_val_batches,
        )
                
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.step_bar is None:
            return

        self.step_bar.reset(self.total_train_batches+self.total_val_batches)
        self.step_bar.set_description_str('Epoch (train)')

    def on_train_batch_end(self, trainer, pl_module, outputs,
                           batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs,
                                   batch, batch_idx, dataloader_idx)
        if self.step_bar is None:
            return
        
        loss = trainer.progress_bar_dict['loss']
        self.step_bar.set_postfix_str(f'loss={loss:0.3f}')
        self.step_bar.update()

    def on_validation_epoch_start(self, trainer, pl_module):
        super().on_validation_epoch_start(trainer, pl_module)
        if self.step_bar is None:
            return

        self.step_bar.set_description_str('Epoch (validation)')

    def on_validation_batch_end(self, trainer, pl_module, outputs,
                                batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs,
                                        batch, batch_idx, dataloader_idx)
        if self.step_bar is None:
            return
        
        self.step_bar.update()

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.epoch_bar is None:
            return

        train_loss = trainer.callback_metrics['train_loss'].item()
        val_loss = trainer.callback_metrics['val_loss'].item()
        self.epoch_bar.set_postfix_str(
            f'train_loss={train_loss:0.3f}, val_loss={val_loss:0.3f})',
            refresh=False
        )
        self.epoch_bar.update()

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        self.epoch_bar.close()
        self.step_bar.close()

    def on_sanity_check_start(self, trainer, pl_module):
        super().on_sanity_check_start(trainer, pl_module)

        print('Sanity check ...', end=' ', file=sys.stderr)

    def on_sanity_check_end(self, trainer, pl_module):
        super().on_sanity_check_end(trainer, pl_module)

        print('Done!', file=sys.stderr)

    # def _update_bar(self, bar: Optional[tqdm], trainer, pl_module, outputs,
    #                 batch, batch_idx, dataloader_idx):
    #     """ Updates the bar by the refresh rate without overshooting. 
    #     taken from pytorch-lightining ProgressBar"""
    #     if bar is None:
    #         return

    #     if bar.total is not None:
    #         delta = min(self.refresh_rate, bar.total - bar.n)
    #     else:
    #         # infinite / unknown size
    #         delta = self.refresh_rate

    #     if delta > 0:
    #         bar.update(delta)
from typing import Optional

from pytorch_lightning.callbacks import ProgressBarBase
from tqdm import tqdm

class EpochAndStepProgressBar(ProgressBarBase):

    def __init__(self, refresh_rate: int = 100):
        super().__init__()
        self.batch_bar = None
        self.epoch_bar = None
        self.refresh_rate = refresh_rate
    #     self.enable = True

    # def disable(self):
    #     self.enable = False

    def on_train_start(self, trainer, pl_module):
        self.on_train_epoch_start(trainer, pl_module)
        self.epoch_bar = tqdm(
            desc='Training', unit='epoch', leave=False,
            dynamic_ncols=True, total=trainer.max_epochs,
        )
        self.batch_bar = tqdm(
            desc='Epoch', unit='step', leave=False,
            dynamic_ncols=True, total=self.total_train_batches,
        )
        # print()
        
    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.batch_bar:
            self.batch_bar.reset(self.total_train_batches)

    def on_train_batch_end(self, trainer, pl_module, outputs,
                           batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs,
                                   batch, batch_idx, dataloader_idx)
        loss = trainer.progress_bar_dict['loss']
        self.batch_bar.set_postfix_str(f'loss={loss}')
        self._update_bar(self.batch_bar)
    
    def on_train_end(self, *args, **kwargs):
        super().on_train_end(*args, **kwargs)
        self.batch_bar.close()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epoch_bar:
            train_loss = trainer.callback_metrics['train_loss'].item()
            val_loss = trainer.callback_metrics['val_loss'].item()
            self.epoch_bar.set_postfix_str(
                f'train_loss={train_loss:0.3f}, val_loss={val_loss:0.3f})'
            )
            self.epoch_bar.update()

    def _update_bar(self, bar: Optional[tqdm]) -> None:
        """ Updates the bar by the refresh rate without overshooting. """
        if bar is None:
            return
        if bar.total is not None:
            delta = min(self.refresh_rate, bar.total - bar.n)
        else:
            # infinite / unknown size
            delta = self.refresh_rate
        if delta > 0:
            bar.update(delta)
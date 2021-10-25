__all__ = ["PrintingCallback", "TqdmCallback"]

from typing import Dict, List

from ray.util.sgd.v2 import SGDCallback
from tqdm import tqdm

class PrintingCallback(SGDCallback):
    def handle_result(self, results: List[Dict], **info):
        print(results)


class TqdmCallback(SGDCallback):
    def __init__(self, max_epochs: int):
        print("hello!", flush=True)
        self.max_epochs = max_epochs
        super().__init__()

    def handle_result(self, results: List[Dict], **info):
        train_loss = sum([result["train_loss"] for result in results]) / len(results)
        val_loss = sum([result["val_loss"] for result in results]) / len(results)

        self.epoch_bar.set_postfix_str(
            f"train_loss={train_loss:0.3f} | val_loss={val_loss:0.3f} "
        )
        self.epoch_bar.update()

    def start_training(self, logdir: str, **info):
        self.epoch_bar = tqdm(
            desc="Training",
            unit="epoch",
            leave=True,
            dynamic_ncols=True,
            total=self.max_epochs,
        )
    
    def finish_training(self, error: bool = False, **info):
        self.epoch_bar.close()

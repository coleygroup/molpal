__all__ = ["PrintingCallback", "TqdmCallback"]

from typing import Dict, List

from ray.train import TrainingCallback
from tqdm import tqdm


class EarlyStoppingCallback(TrainingCallback):
    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 10,
        minimize: bool = True,
        verbose: bool = False,
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait_count = 0
        self.minimize = minimize
        self.verbose = verbose

        super().__init__()

    def handle_result(self, results: List[Dict], **info):
        avg_result = sum(results[self.monitor]) / len(results)

        delta = avg_result - self.curr_best
        delta = -1 * delta if self.minimize else delta

        if delta > self.min_delta:
            self.curr_best = avg_result
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count > self.patience:
            print("STOP") if self.verbose else None

    def start_training(self, logdir: str, **info):
        self.curr_best = float("inf") if self.minimize else float("-inf")


class PrintingCallback(TrainingCallback):
    def handle_result(self, results: List[Dict], **info):
        print(results)


class TqdmCallback(TrainingCallback):
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs
        super().__init__()

    def handle_result(self, results: List[Dict], **info):
        train_loss = sum([result["train_loss"] for result in results]) / len(results)
        val_loss = sum([result["val_loss"] for result in results]) / len(results)

        self.epoch_bar.set_postfix_str(f"train_loss={train_loss:0.3f} | val_loss={val_loss:0.3f} ")
        self.epoch_bar.update()

    def start_training(self, logdir: str, **info):
        self.epoch_bar = tqdm(
            desc="Training", unit="epoch", leave=True, dynamic_ncols=True, total=self.max_epochs
        )

    def finish_training(self, error: bool = False, **info):
        self.epoch_bar.close()

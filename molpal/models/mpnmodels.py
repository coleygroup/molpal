"""This module contains Model implementations that utilize the MPNN model as their underlying
model"""
from __future__ import annotations

from functools import partial
import json
import logging
from pathlib import Path
from typing import Iterable, List, NoReturn, Optional, Sequence, Tuple
import warnings

import numpy as np
from pytorch_lightning import Trainer as PlTrainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import ray
from ray.train import Trainer as RayTrainer
import torch
from tqdm import tqdm

from molpal.utils import batches
from molpal.models import mpnn
from molpal.models.base import Model
from molpal.models.chemprop.data.data import MoleculeDatapoint, MoleculeDataset, MoleculeDataLoader
from molpal.models.chemprop.data.scaler import StandardScaler
from molpal.models.chemprop.data.utils import split_data

logging.getLogger("lightning").setLevel(logging.FATAL)
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


class MPNN:
    """A message-passing neural network wrapper class that uses Chemprop as the underlying model

    Attributes
    ----------
    ncpu : int
        the number of cores over which to parallelize input batch preparation
    ddp : bool
        whether to train the model over a distributed setup. Only works with CUDA >= 11.0
    precision : int
        the bit precision with which to train the model
    model : MoleculeModel
        the underlying chemprop model on which to train and make predictions
    uncertainty : Optional[str], default=None
        the uncertainty quantification method the model uses. None if it does not use any
        uncertainty quantification
    loss_func : Callable
        the loss function used in model training
    batch_size : int
        the size of each minibatch during training
    epochs : int
        the number of epochs over which to train
    dataset_type : str
        the type of dataset. Choices: ('regression')
        TODO: add support for classification
    num_tasks : int
        the number of training tasks
    use_gpu : bool
        whether the GPU will be used.
        NOTE: If a GPU is detected, it will be used. If this is undesired, set the
        CUDA_VISIBLE_DEVICES environment variable to be empty
    num_workers : int
        the number of workers to distribute model training over. Equal to the number of GPUs
        detected, or if none are available, the ratio of total CPUs on detected on the ray cluster
        over the number of CPUs to dedicate to each dataloader
    train_config : Dict
        a dictionary containing the configuration of training variables: learning rates, maximum
        epochs, validation metric, etc.
    scaler : StandardScaler
        a scaler to normalize target data before training and validation and to reverse transform
        prediction outputs
    """

    def __init__(
        self,
        batch_size: int = 50,
        uncertainty: Optional[str] = None,
        dataset_type: str = "regression",
        num_tasks: int = 1,
        atom_messages: bool = False,
        hidden_size: int = 300,
        bias: bool = False,
        depth: int = 3,
        dropout: float = 0.0,
        undirected: bool = False,
        activation: str = "ReLU",
        ffn_hidden_size: Optional[int] = None,
        ffn_num_layers: int = 2,
        metric: str = "rmse",
        epochs: int = 50,
        warmup_epochs: float = 2.0,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        ncpu: int = 1,
        ddp: bool = False,
        precision: int = 32,
        model_seed: Optional[int] = None,
    ):
        self.ncpu = ncpu
        self.ddp = ddp
        if precision not in (16, 32):
            raise ValueError(f'arg: "precision" can only be (16, 32)! got: {precision}')
        self.precision = precision

        self.model = mpnn.MoleculeModel(
            uncertainty,
            dataset_type,
            num_tasks,
            atom_messages,
            bias,
            depth,
            dropout,
            undirected,
            activation=activation,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            ffn_num_layers=ffn_num_layers,
        )

        self.uncertainty = uncertainty
        self.dataset_type = dataset_type
        self.num_tasks = num_tasks

        self.epochs = epochs
        self.batch_size = batch_size

        self.scaler = None

        ngpu = int(ray.cluster_resources().get("GPU", 0))
        if ngpu > 0:
            self.use_gpu = True
            self._predict = mpnn.predict_.options(num_cpus=ncpu, num_gpus=1)
            self.num_workers = ngpu
        else:
            self.use_gpu = False
            self._predict = mpnn.predict_.options(num_cpus=ncpu)
            self.num_workers = int(ray.cluster_resources()["CPU"] // self.ncpu)

        self.seed = model_seed
        if model_seed is not None:
            torch.manual_seed(model_seed)

        self.train_config = {
            "model": self.model,
            "uncertainty": self.uncertainty,
            "dataset_type": dataset_type,
            "batch_size": self.batch_size,
            "warmup_epochs": warmup_epochs,
            "max_epochs": self.epochs,
            "init_lr": init_lr,
            "max_lr": max_lr,
            "final_lr": final_lr,
            "metric": metric,
        }

    def train(self, smis: Iterable[str], targets: np.ndarray) -> bool:
        """Train the model on the inputs SMILES with the given targets"""
        train_data, val_data = self.make_datasets(smis, targets)

        if self.ddp:
            self.train_config["train_data"] = train_data
            self.train_config["val_data"] = val_data

            callbacks = [mpnn.ray.TqdmCallback(self.epochs)]
            trainer = RayTrainer("torch", self.num_workers, self.use_gpu, {"CPU": self.ncpu})
            trainer.start()
            results = trainer.run(mpnn.ray.train_func, self.train_config, callbacks)
            trainer.shutdown()

            self.model = results[0]

            return True

        train_dataloader = MoleculeDataLoader(
            train_data, self.batch_size, self.ncpu, pin_memory=False
        )
        val_dataloader = MoleculeDataLoader(val_data, self.batch_size, self.ncpu, pin_memory=False)

        lit_model = mpnn.ptl.LitMPNN(self.train_config)

        callbacks = [
            EarlyStopping("val_loss", patience=10, mode="min"),
            mpnn.ptl.EpochAndStepProgressBar(),
        ]
        trainer = PlTrainer(
            logger=False,
            max_epochs=self.epochs,
            callbacks=callbacks,
            gpus=1 if self.use_gpu else 0,
            precision=self.precision,
            enable_model_summary=False,
            enable_checkpointing=False,  # TODO: reimplement trainer checkpointing later
        )
        trainer.fit(lit_model, train_dataloader, val_dataloader)

        return True

    def make_datasets(
        self, xs: Iterable[str], ys: np.ndarray
    ) -> tuple[MoleculeDataset, MoleculeDataset]:
        """Split xs and ys into train and validation datasets"""
        data = MoleculeDataset([MoleculeDatapoint([x], y) for x, y in zip(xs, ys.reshape(-1, 1))])
        train_data, val_data, _ = split_data(data, "random", (0.8, 0.2, 0.0), seed=self.seed)

        self.scaler = train_data.normalize_targets()
        val_data.scale_targets(self.scaler)

        return train_data, val_data

    def predict(self, smis: Iterable[str]) -> np.ndarray:
        """Generate predictions for the inputs xs

        Parameters
        ----------
        smis : Iterable[str]
            the SMILES strings for which to generate predictions

        Returns
        -------
        np.ndarray
            an array of shape `n x m`, where `n` is the number of SMILES strings and `m` is the
            number of tasks
        """
        model = ray.put(self.model)
        scaler = ray.put(self.scaler)

        refs = [
            self._predict.remote(
                model,
                smis,
                self.batch_size,
                self.ncpu,
                self.uncertainty,
                scaler,
                self.use_gpu,
                True,
            )
            for smis in batches(smis, 20000)
        ]
        Y_pred_batches = [ray.get(r) for r in tqdm(refs, "Prediction", unit="batch", leave=False)]
        Y_pred = np.concatenate(Y_pred_batches)

        if self.scaler is not None:
            if self.uncertainty == "mve":
                Y_pred[:, 0::2] = Y_pred[:, 0::2] * self.scaler.stds + self.scaler.means
                Y_pred[:, 1::2] *= self.scaler.stds**2
            else:
                Y_pred = Y_pred * self.scaler.stds + self.scaler.means

        return Y_pred

    def save(self, path) -> str:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_path = f"{path}/model.pt"
        torch.save(self.model.state_dict(), model_path)

        state_path = f"{path}/state.json"
        try:
            state = {
                "model_path": model_path,
                "means": self.scaler.means.tolist(),
                "stds": self.scaler.stds.tolist(),
            }
        except AttributeError:
            state = {"model_path": model_path}

        json.dump(state, open(state_path, "w"), indent=4)

        return state_path

    def load(self, path):
        state = json.load(open(path, "r"))

        self.model.load_state_dict(torch.load(state["model_path"]))
        try:
            self.scaler = StandardScaler(state["means"], state["stds"])
        except KeyError:
            pass


class MPNModel(Model):
    """Message-passing model that learns feature representations of inputs and
    passes these inputs to a feed-forward neural network to predict means"""

    def __init__(
        self,
        test_batch_size: Optional[int] = 1000000,
        ncpu: int = 1,
        ddp: bool = False,
        precision: int = 32,
        model_seed: Optional[int] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 1000000

        self.build_model = partial(
            MPNN, ncpu=ncpu, ddp=ddp, precision=precision, model_seed=model_seed
        )
        self.model = self.build_model()

        super().__init__(test_batch_size, **kwargs)

    @property
    def provides(self):
        return {"means"}

    @property
    def type_(self):
        return "mpn"

    def train(self, xs: Iterable[str], ys: np.ndarray, *, retrain: bool = False, **kwargs) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys)

    def get_means(self, xs: Sequence[str]) -> np.ndarray:
        preds = self.model.predict(xs)
        return preds[:, 0]  # assume single-task

    def get_means_and_vars(self, xs: List) -> NoReturn:
        raise TypeError("MPNModel cannot predict variance!")

    def save(self, path) -> str:
        return self.model.save(path)

    def load(self, path):
        self.model.load(path)


class MPNDropoutModel(Model):
    """Message-passing network model that predicts means and variances through
    stochastic dropout during model inference"""

    def __init__(
        self,
        test_batch_size: Optional[int] = 1000000,
        dropout: float = 0.2,
        dropout_size: int = 10,
        ncpu: int = 1,
        ddp: bool = False,
        precision: int = 32,
        model_seed: Optional[int] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 1000000

        self.build_model = partial(
            MPNN,
            uncertainty="dropout",
            dropout=dropout,
            ncpu=ncpu,
            ddp=ddp,
            precision=precision,
            model_seed=model_seed,
        )
        self.model = self.build_model()

        self.dropout_size = dropout_size

        super().__init__(test_batch_size, **kwargs)

    @property
    def type_(self):
        return "mpn"

    @property
    def provides(self):
        return {"means", "vars", "stochastic"}

    def train(self, xs: Iterable[str], ys: np.ndarray, *, retrain: bool = False, **kwargs) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys)

    def get_means(self, xs: Sequence[str]) -> np.ndarray:
        predss = self._get_predictions(xs)
        return np.mean(predss, axis=1)

    def get_means_and_vars(self, xs: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        predss = self._get_predictions(xs)

        return predss.mean(1), predss.var(1)

    def _get_predictions(self, xs: Sequence[str]) -> np.ndarray:
        predss = np.zeros((len(xs), self.dropout_size))
        for j in tqdm(range(self.dropout_size), "Dropout", unit="pass"):
            predss[:, j] = self.model.predict(xs)[:, 0]  # assume single-task

        return predss

    def save(self, path) -> str:
        return self.model.save(path)

    def load(self, path):
        self.model.load(path)


class MPNTwoOutputModel(Model):
    """Message-passing network model that predicts means and variances through mean-variance
    estimation"""

    def __init__(
        self,
        test_batch_size: Optional[int] = 1000000,
        ncpu: int = 1,
        ddp: bool = False,
        precision: int = 32,
        model_seed: Optional[int] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 1000000

        self.build_model = partial(
            MPNN, uncertainty="mve", ncpu=ncpu, ddp=ddp, precision=precision, model_seed=model_seed
        )
        self.model = self.build_model()

        super().__init__(test_batch_size, **kwargs)

    @property
    def type_(self):
        return "mpn"

    @property
    def provides(self):
        return {"means", "vars"}

    def train(self, xs: Iterable[str], ys: np.ndarray, *, retrain: bool = False, **kwargs) -> bool:
        if retrain:
            self.model = self.build_model()

        return self.model.train(xs, ys)

    def get_means(self, xs: Sequence[str]) -> np.ndarray:
        means, _ = self._get_predictions(xs)
        return means

    def get_means_and_vars(self, xs: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        means, variances = self._get_predictions(xs)
        return means, variances

    def _get_predictions(self, xs: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Get both the means and the variances for the xs"""
        preds = self.model.predict(xs)
        # assume single task prediction now
        # means, variances = preds[:, 0::2], preds[:, 1::2]
        means, variances = preds[:, 0], preds[:, 1]  #
        return means, variances

    def save(self, path) -> str:
        return self.model.save(path)

    def load(self, path):
        self.model.load(path)


# def combine_sds(sd1: float, mu1: float, n1: int,
#                 sd2: float, mu2: float, n2: int):

#     var1 = sd1**2
#     var2 = sd2**2
#     n_total = n1 + n2
#     mu_combined = (n1*mu1 + n2*mu2) / n_total

#     sd_combined = sqrt(
#         (n1*(var1 + (mu1-mu_combined)**2) + n2*(var2 + (mu2-mu_combined)**2))
#         / n_total
#     )
#     return sd_combined

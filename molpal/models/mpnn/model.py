from argparse import Namespace
from typing import Optional

import torch
from torch import nn

from molpal.models.chemprop.models.mpn import MPN
from molpal.models.chemprop.nn_utils import get_activation_function, initialize_weights
from molpal.models.chemprop.utils import UncertaintyType


class EvaluationDropout(nn.Dropout):
    def forward(self, input):
        return nn.functional.dropout(input, p=self.p)


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network
    followed by a feed-forward neural network.

    Attributes
    ----------
    uncertainty : Optional[str]
        the uncertainty method this model is using, if any
    classification : bool
        whether this model is a classification model
    output_size : int
        the size of the output layer for the feed-forward network
    encoder : MPN
        the message-passing encoder of the message-passing network
    ffn : nn.Sequential
        the feed-forward network of the message-passing network
    """

    def __init__(
        self,
        uncertainty: Optional[str] = None,
        dataset_type: str = "regression",
        num_tasks: int = 1,
        atom_messages: bool = False,
        bias: bool = False,
        depth: int = 3,
        dropout: float = 0.0,
        undirected: bool = False,
        aggregation: str = "mean",
        aggregation_norm: int = 100,
        activation: str = "ReLU",
        hidden_size: int = 300,
        ffn_hidden_size: Optional[int] = None,
        ffn_num_layers: int = 2,
    ):
        super().__init__()

        if uncertainty is not None:
            self.uncertainty = {"dropout": UncertaintyType.DROPOUT, "mve": UncertaintyType.MVE}[
                uncertainty.lower()
            ]
        else:
            self.uncertainty = None

        self.classification = dataset_type.lower() == "classification"
        if self.classification:
            self.sigmoid = nn.Sigmoid()

        self.output_size = num_tasks

        self.encoder = self.build_encoder(
            atom_messages,
            bias,
            hidden_size,
            depth,
            dropout,
            undirected,
            aggregation,
            aggregation_norm,
            activation,
        )
        self.ffn = self.build_ffn(
            num_tasks, hidden_size, dropout, activation, ffn_num_layers, ffn_hidden_size
        )

        initialize_weights(self)

    def build_encoder(
        self,
        atom_messages: bool = False,
        bias: bool = False,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        undirected: bool = False,
        aggregation: str = "mean",
        aggregation_norm: int = 100,
        activation: str = "ReLU",
    ):
        return MPN(
            Namespace(
                atom_messages=atom_messages,
                hidden_size=hidden_size,
                bias=bias,
                depth=depth,
                dropout=dropout,
                undirected=undirected,
                features_only=False,
                use_input_features=False,
                aggregation=aggregation,
                aggregation_norm=aggregation_norm,
                activation=activation,
                number_of_molecules=1,
                atom_descriptors=None,
                mpn_shared=False,
            )
        )

    def build_ffn(
        self,
        output_size: int,
        hidden_size: int = 300,
        dropout: float = 0.0,
        activation: str = "ReLU",
        ffn_num_layers: int = 2,
        ffn_hidden_size: Optional[int] = None,
    ) -> None:
        first_linear_dim = hidden_size

        if self.uncertainty == UncertaintyType.DROPOUT:
            dropout = EvaluationDropout(dropout)
        else:
            dropout = nn.Dropout(dropout)

        activation = get_activation_function(activation)

        if self.uncertainty == UncertaintyType.MVE:
            output_size *= 2

        if ffn_num_layers == 1:
            ffn = [dropout, nn.Linear(first_linear_dim, output_size)]
        else:
            if ffn_hidden_size is None:
                ffn_hidden_size = hidden_size

            ffn = [dropout, nn.Linear(first_linear_dim, ffn_hidden_size)]
            for _ in range(ffn_num_layers - 2):
                ffn.extend([activation, dropout, nn.Linear(ffn_hidden_size, ffn_hidden_size)])
            ffn.extend([activation, dropout, nn.Linear(ffn_hidden_size, output_size)])

        return nn.Sequential(*ffn)

    def featurize(self, *inputs):
        """Compute feature vectors of the input."""
        return self.ffn[:-1](self.encoder(*inputs))

    def forward(self, *inputs):
        """Runs the MoleculeModel on the input."""
        z = self.ffn(self.encoder(*inputs))

        if self.uncertainty == UncertaintyType.MVE:
            pred_vars = z[:, 1::2]
            capped_vars = nn.functional.softplus(pred_vars)

            z = torch.clone(z)
            z[:, 1::2] = capped_vars

        if self.classification and not self.training:
            z = self.sigmoid(z)

        # if self.multiclass:
        #     # batch size x num targets x num classes per target
        #     output = output.reshape((output.size(0), -1, self.num_classes))
        #     if not self.training:
        #         # to get probabilities during evaluation, but not during
        #         # training as we're using CrossEntropyLoss
        #         output = self.multiclass_softmax(output)

        return z

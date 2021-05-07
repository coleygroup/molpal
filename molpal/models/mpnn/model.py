from argparse import Namespace
from typing import Optional

import torch
from torch import nn

from ..chemprop.models.mpn import MPN
from ..chemprop.nn_utils import get_activation_function, initialize_weights

class EvaluationDropout(nn.Dropout):
    def forward(self, input):
        return nn.functional.dropout(input, p = self.p)


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network
    followed by a feed-forward neural network.

    Attributes
    ----------
    uncertainty : Optional[str]
        the uncertainty method this model uses. Choices include:
        - 'dropout': performs dropout during inference. Returns only one 
            prediction for each task during inference. Must manually perform 
            multiple forward passes to generate an uncertainty estimate
        - 'mve': use mean-variance estimation to learn uncertainty. Output
            size is equal to 2*num_tasks, where the predicted means are indices
            0::2 and the predicted variances are indices 1::2
        - 'evidential': use evidential uncertainty estimation to learn 
            uncertainty. Output size is equal to 4*num_tasks, where each task
            has 4 outputs associated with it: mean, lambda, alpha, and beta.
            I.e., indices 0::4, 1::4, 2::4, and 3::4
    uncertainty : bool
        whether this model predicts its own uncertainty values
        (e.g. Mean-Variance estimation)
    classification : bool
        whether this model is a classification model
    output_size : int
        the size of the output layer for the feed-forward network
    encoder : MPN
        the message-passing encoder of the message-passing network
    ffn : nn.Sequential
        the feed-forward network of the message-passing network
    """
    def __init__(self,
                 uncertainty: Optional[str] = None,
                 dataset_type: str = 'regression', num_tasks: int = 1,
                 atom_messages: bool = False, bias: bool = False,
                 depth: int = 3, dropout: float = 0.0, undirected: bool = False,
                 aggregation: str = 'mean', aggregation_norm: int = 100,
                 activation: str = 'ReLU', hidden_size: int = 300, 
                 ffn_hidden_size: Optional[int] = None,
                 ffn_num_layers: int = 2):
        super().__init__()

        # self.uncertainty_method = uncertainty_method
        self.uncertainty = uncertainty
        self.classification = dataset_type == 'classification'

        if self.classification:
            if self.uncertainty != 'evidential':
                self.sigmoid = nn.Sigmoid()
            else:
                self.sigmoid = nn.Identity()

        self.encoder = self.build_encoder(
            atom_messages=atom_messages, hidden_size=hidden_size,
            bias=bias, depth=depth, dropout=dropout, undirected=undirected,
            aggregation=aggregation, aggregation_norm=aggregation_norm,
            activation=activation
        )
        self.ffn = self.build_ffn(
            output_size=num_tasks, hidden_size=hidden_size, dropout=dropout, 
            activation=activation, ffn_num_layers=ffn_num_layers, 
            ffn_hidden_size=ffn_hidden_size
        )

        initialize_weights(self)
    
    def build_encoder(self, atom_messages: bool = False, bias: bool = False,
                      hidden_size: int = 300, depth: int = 3,
                      dropout: float = 0.0, undirected: bool = False,
                      aggregation: str = 'mean', aggregation_norm: int = 100,
                      activation: str = 'ReLU'):
         return MPN(Namespace(
            atom_messages=atom_messages, hidden_size=hidden_size,
            bias=bias, depth=depth, dropout=dropout, undirected=undirected,
            features_only=False, use_input_features=False,
            aggregation=aggregation, aggregation_norm=aggregation_norm,
            activation=activation, number_of_molecules=1,
            atom_descriptors=None, mpn_shared=False
        ))

    def build_ffn(self, output_size: int, 
                  hidden_size: int = 300, dropout: float = 0.0,
                  activation: str = 'ReLU', ffn_num_layers: int = 2,
                  ffn_hidden_size: Optional[int] = None) -> None:
        first_linear_dim = hidden_size

        # If dropout uncertainty method, use for both evaluation and training
        if self.uncertainty == 'dropout':
            dropout = EvaluationDropout(dropout)
        else:
            dropout = nn.Dropout(dropout)

        activation = get_activation_function(activation)

        if self.uncertainty == 'mve':
            output_size *= 2
        elif self.uncertainty == 'evidential':
            if self.classiciation:
                output_size *= 2
            else:
                output_size *= 4

        if ffn_num_layers == 1:
            ffn = [dropout,
                   nn.Linear(first_linear_dim, output_size)]
        else:
            if ffn_hidden_size is None:
                ffn_hidden_size = hidden_size

            ffn = [dropout,
                   nn.Linear(first_linear_dim, ffn_hidden_size)]
            for _ in range(ffn_num_layers - 2):
                ffn.extend([activation,
                            dropout,
                            nn.Linear(ffn_hidden_size, ffn_hidden_size)])
            ffn.extend([activation,
                        dropout,
                        nn.Linear(ffn_hidden_size, output_size)])

        return nn.Sequential(*ffn)

    def featurize(self, *inputs):
        """Compute feature vectors of the input."""
        return self.ffn[:-1](self.encoder(*inputs))

    def forward(self, *inputs):
        """Runs the MoleculeModel on the input."""
        z = self.ffn(self.encoder(*inputs))

        if self.uncertainty == 'mve':
            means = z[:, 0::2]
            variances = z[:, 1::2]

            capped_variances = nn.functional.softplus(variances)

            z = torch.clone(z)
            z[:, 1::2] = capped_variances
            # z = torch.stack((means, capped_variances), dim=2).view(z.size())

        if self.uncertainty == 'evidence':
            if self.classification:
                z = nn.functional.softplus(z) + 1
            else:
                min_val = 1e-6

                means = z[:, 0::4]
                loglambdas = z[:, 1::4]
                logalphas = z[:, 2::4]
                logbetas = z[:, 3::4]

                lambdas = nn.functional.softplus(loglambdas) + min_val
                alphas = nn.functional.softplus(logalphas) + min_val + 1
                betas = nn.functional.softplus(logbetas) + min_val

                z = torch.clone(z)
                z[:, 1::4] = lambdas
                z[:, 2::4] = alphas
                z[:, 3::4] = betas

                # z = torch.stack((means, lambdas, alphas, betas),
                #                 dim = 2).view(z.size())
        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
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

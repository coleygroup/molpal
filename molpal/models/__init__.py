"""This module contains the Model ABC and various implementations thereof. A
model is used to predict an input's objective function based on prior
training data."""

from typing import Optional, Type

from molpal.models.base import Model
from molpal.models.utils import get_model_types
from molpal.models.nnmodels import (NNModel, NNDropoutModel,
                                    NNEnsembleModel, NNTwoOutputModel)
from molpal.models.mpnmodels import (MPNModel, MPNDropoutModel, 
                                     MPNTwoOutputModel)
from molpal.models.sklmodels import RFModel, GPModel

def model(model: str, **kwargs) -> Type[Model]:
    """Model factory function"""
    try:
        return {
            'rf': RFModel,
            'gp': GPModel,
            'nn': nn,
            'mpn': mpn
        }[model](**kwargs)
    except KeyError:
        raise NotImplementedError(f'Unrecognized model: "{model}"')

def nn(conf_method: Optional[str] = None, **kwargs) -> Type[Model]:
    """NN-type Model factory function"""
    if conf_method is None:
        return NNModel(**kwargs)

    try:
        return {
            'dropout': NNDropoutModel,
            'ensemble': NNEnsembleModel,
            'twooutput': NNTwoOutputModel,
            'mve': NNTwoOutputModel,
            'none': NNModel
        }[conf_method](conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(
            f'Unrecognized NN confidence method: "{conf_method}"')

def mpn(conf_method: Optional[str] = None, **kwargs) -> Type[Model]:
    """MPN-type Model factory function"""
    if conf_method is None:
        return MPNModel(**kwargs)
        
    try:
        return {
            'dropout': MPNDropoutModel,
            'twooutput': MPNTwoOutputModel,
            'mve': MPNTwoOutputModel,
            'none': MPNModel
        }[conf_method](conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(
            f'Unrecognized MPN confidence method: "{conf_method}"')

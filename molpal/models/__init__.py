from typing import Optional, Type

from .base import Model
from .utils import get_model_types
from .nnmodels import NNModel, NNDropoutModel, NNEnsembleModel, NNTwoOutputModel
from .mpnmodels import MPNModel, MPNDropoutModel, MPNTwoOutputModel
from .sklmodels import RFModel, GPModel

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

import torch
import torch.nn as nn
from ..modules import *

bayes_layer = (BayesLinear, BayesConv2d, BayesBatchNorm2d)  

def freeze(module):
    """
    Methods for freezing bayesian-model.

    Arguments:
        model (nn.Module): a model to be freezed.

    """

    if isinstance(module, bayes_layer) :
        module.freeze()
    for submodule in module.children() :
        freeze(submodule)
        

def unfreeze(module):
    """
    Methods for unfreezing bayesian-model.

    Arguments:
        model (nn.Module): a model to be unfreezed.

    """
    if isinstance(module, bayes_layer) :
        module.unfreeze()
    for submodule in module.children() :
        unfreeze(submodule)
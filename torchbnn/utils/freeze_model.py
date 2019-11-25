import torch
import torch.nn as nn
from ..modules import *

bayes_layer = (BayesLinear, BayesConv2d, BayesBatchNorm2d)  

"""
Methods for freezing bayesian-model.

Arguments:
    model (nn.Module): a model to be freezed.

"""

def freeze(module):
    if isinstance(module, bayes_layer) :
        module.freeze()
    for submodule in module.children() :
        freeze(submodule)
        
"""
Methods for unfreezing bayesian-model.

Arguments:
    model (nn.Module): a model to be freezed.

"""

def unfreeze(module):
    if isinstance(module, bayes_layer) :
        module.unfreeze()
    for submodule in module.children() :
        unfreeze(submodule)
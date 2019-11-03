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
        module.freeze = True
    for submodule in module.children() :
        freeze(submodule)
        
def unfreeze(module):
    if isinstance(module, bayes_layer) :
        module.freeze = False
    for submodule in module.children() :
        unfreeze(submodule)
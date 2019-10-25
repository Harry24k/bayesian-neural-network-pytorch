import torch
import torch.nn as nn
from ..modules import *

bayes_layer = [BayesLinear, BayesConv2d, BayesBatchNorm2d]    

"""
Methods for freezing bayesian-layer.

Arguments:
    layer (nn.Module): a layer to be freezed.

"""  

def _freeze(layer):
    for inst in bayes_layer :
        if isinstance(layer, inst) :
            layer.freeze = True
        else :
            continue

def _unfreeze(layer):
    for inst in bayes_layer :
        if isinstance(layer, inst) :
            layer.freeze = False
        else :
            continue
    return layer

"""
Methods for freezing bayesian-model.

Arguments:
    model (nn.Module): a model to be freezed.

"""

def freeze(model):
    for name, m in model.named_children() :
        if isinstance(m, nn.Sequential) :
            for layer in m :
                _freeze(layer)
        else :
            _freeze(layer)
            
def unfreeze(model):
    for name, m in model.named_children() :
        if isinstance(m, nn.Sequential) :
            for layer in m :
                _unfreeze(layer)
        else :
            _unfreeze(layer)
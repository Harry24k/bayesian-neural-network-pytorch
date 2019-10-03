import copy

import torch
from torch.nn import *
from ..modules import *

nonbayes_layer = [Linear, Conv2d, BatchNorm2d]
bayes_layer = [BayesLinear, BayesConv2d, BayesBatchNorm2d]    

"""
Methods for the transformation between non-bayesian-layer and bayesian-layer.

Arguments:
    layer (nn.Module): a layer to be transformed.
    prior_mu (Float): mean of prior normal distribution.
    prior_log_sigma (Float): log(sigma of prior normal distribution).

"""

def _nonbayes_to_bayes(layer, prior_mu, prior_sigma):
    for inst in nonbayes_layer :
        if isinstance(layer, inst) :
            # Can't find a way using layer.__dict__
            # Instead of this, using exec
            s = str(layer).split("(", 1)
            s = "Bayes" +  s[0] + "(" + str(prior_mu) + ", " + str(prior_sigma) + ", " + s[1]
            s = "bayeslayer = " + s
            exec(s, globals())
            return bayeslayer
        else :
            continue
    return layer    

def _bayes_to_nonbayes(layer):
    for inst in bayes_layer :
        if isinstance(layer, inst) :
            # Can't find a way using layer.__dict__
            # Instead of this, using exec
            s = str(layer)[5:].split("(", 1)
            s = s[0] + "(" + s[1].split(',', 2)[-1]
            s = "nonbayeslayer = " + s
            exec(s, globals())
            return nonbayeslayer
        else :
            continue
    return layer

"""
Methods for the transformation between non-bayesian-model and bayesian-model.

Arguments:
    model (nn.Module): a model to be transformed.
    prior_mu (Float): mean of prior normal distribution.
    prior_sigma (Float): sigma of prior normal distribution.

"""

def nonbayes_to_bayes(local_model, prior_mu, prior_sigma, inplace=True):
    global model
    
    if inplace :
        model = local_model
    else :
        model = copy.deepcopy(local_model)
        
    for name, m in model.named_children() :
        if isinstance(m, Sequential) :
            s = "model."+ name + " = Sequential("
            for layer in m :
                layer = _nonbayes_to_bayes(layer, prior_mu, prior_sigma)
                s += str(layer) + ", "
            s += ")"
            exec(s, globals())
        else :
            layer = _nonbayes_to_bayes(m, prior_mu, prior_sigma)
            s = "model."+ name + " = " + str(layer)
            exec(s, globals())
    return model

def bayes_to_nonbayes(local_model, inplace=True):
    global model
    
    if inplace :
        model = local_model
    else :
        model = copy.deepcopy(local_model)
        
    for name, m in model.named_children() :
        if isinstance(m, Sequential) :
            s = "model."+ name + " = Sequential("
            for layer in m :
                layer = _bayes_to_nonbayes(layer)
                s += str(layer) + ", "
            s += ")"
            exec(s, globals())
        else :
            layer = _bayes_to_nonbayes(m)
            s = "model."+ name + " = " + str(layer)
            exec(s, globals())
    return model
    

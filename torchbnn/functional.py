import math
import torch

from .modules import *

def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1) :
    """
    An method for calculating KL divergence between two Normal distribtuion.

    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).
   
    """
    kl = log_sigma_1 - log_sigma_0 + \
    (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*math.exp(log_sigma_1)**2) - 0.5
    return kl.sum()

def bayesian_kl_loss(model, norm=True, last_layer_only=False) :
    """
    An method for calculating KL divergence of whole layers in the model.

    Arguments:
        model (nn.Module): a model to be calculated.
        norm (Bool): True return mean of each layer's KL divergence, 
                     False return sum of each layer's KL divergence.  
        last_layer_only (Bool): True for return the last layer's KL divergence.    
        
    """
    kl_sum = 0
    n = 0

    for m in model.modules() :
        if isinstance(m, (BayesLinear, BayesConv2d)):
            kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma)
            kl_sum += kl
            n += len(m.weight_mu.view(-1))

            if m.bias :
                kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl
                n += len(m.bias_mu.view(-1))

        if isinstance(m, BayesBatchNorm2d):
            if m.affine :
                kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl
                n += len(m.weight_mu.view(-1))

                kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl                
                n += len(m.bias_mu.view(-1))

    if norm :
        kl_sum = kl_sum/n

    if last_layer_only :
        kl_sum = kl
        
    return kl_sum



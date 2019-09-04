import math

import torch
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F

"""
Applies Bayesian Linear

Arguments:
    prior_mu (Float): mean of prior normal distribution.
    prior_log_sigma (Float): log(sigma of prior normal distribution).

.. note:: other arguments are following linear of pytorch 1.2.0.
https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py

"""

class BayesLinear(Module):
    __constants__ = ['prior_mu', 'prior_log_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_log_sigma, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.prior_mu = prior_mu
        self.prior_log_sigma = prior_log_sigma
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))
                
        self.bias = bias
            
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        # Pytorch-BayesianCNN
#         stdv = 1. / math.sqrt(self.weight_mu.size(1))
#         self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        
        if self.bias :
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_mu, -bound, bound)
            
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def forward(self, input):
        weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        
        if self.bias:
            bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
        else :
            bias = None
            
        return F.linear(input, weight, bias)

    def extra_repr(self):
        return 'prior_mu={}, prior_log_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu, self.prior_log_sigma, self.in_features, self.out_features, self.bias is not None)
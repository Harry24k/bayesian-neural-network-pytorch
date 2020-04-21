import torch
from torch.nn import Module


class BayesModule(Module) :
    r"""
    Applies Bayesian Module
    Currently this module is not being used as base of bayesian modules because it has not many utilies yet,
    However, it can be used in the near future for convenience.
    """
    
    def freeze(self):
        r"""Sets the module in freezed mode.
        This has effect on bayesian modules. It will fix epsilons, e.g. weight_eps, bias_eps.
        Thus, bayesian neural networks will return same results with same inputs.
        """
        self.freeze = True
        for module in self.children():
            module.freeze(mode)
        return self
    
    def unfreeze(self):
        r"""Sets the module in unfreezed mode.
        This has effect on bayesian modules. It will unfix epsilons, e.g. weight_eps, bias_eps.
        Thus, bayesian neural networks will return different results even if same inputs are given.
        """
        self.freeze = False
        for module in self.children():
            module.unfreeze(mode)
        return self
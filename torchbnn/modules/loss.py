import warnings

from torch.nn import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

from .. import functional as BF

class _Loss(Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction
            
class BKLLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction='mean', last_layer_only=False):
        super(BKLLoss, self).__init__(reduction)
        self.last_layer_only = last_layer_only

    def forward(self, model):
        return BF.bayesian_kl_loss(model, reduction=self.reduction, last_layer_only=self.last_layer_only)
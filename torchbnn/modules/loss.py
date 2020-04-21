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
    """
    Loss for calculating KL divergence of baysian neural network model.

    Arguments:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.    
    """
    __constants__ = ['reduction']

    def __init__(self, reduction='mean', last_layer_only=False):
        super(BKLLoss, self).__init__(reduction)
        self.last_layer_only = last_layer_only

    def forward(self, model):
        """
        Arguments:
            model (nn.Module): a model to be calculated for KL-divergence.
        """
        return BF.bayesian_kl_loss(model, reduction=self.reduction, last_layer_only=self.last_layer_only)
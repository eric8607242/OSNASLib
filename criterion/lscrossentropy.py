###################################
#    LabelSmoothingCrossEntropy   #
###################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """ Smoothing cross entropy loss
    Reference : https://arxiv.org/pdf/1512.00567.pdf
    """
    def __init__(self, criterion_config, smoothing=0.1):
        """
        Args: criterion_config (dict): The config for criterion
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.criterion_config = criterion_config

    def forward(self, x, target):
        """ Compute smoothing cross entropy loss

        Args:
            x (torch.Tensor): 
            target (torch.Tensor)

        Return:
            loss (torch.Tensor)
        """
        logprobs = F.log_softmax(x, dim=-1)
        
        # Get the predict probability corresponding to target class(index)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()



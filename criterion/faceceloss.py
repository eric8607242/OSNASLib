import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

class FaceCELoss(nn.Module):
    def __init__(self, criterion_config):
        """
        Args: criterion_config (dict): The config for criterion
        """
        super(FaceCELoss, self).__init__()
        self.criterion_config = criterion_config

        embedding_channels = self.criterion_config["embedding_channels"]
        true_classes = self.criterion_config["training_classes"]

        self.identity_weights = nn.Parameter(torch.Tensor(embedding_channels, true_classes))
        nn.init.xavier_uniform_(self.identity_weights)

        self.ce_loss = CrossEntropyLoss()

    def forward(self, x, target):
        """
        Args:
            x (torch.Tensor): 
            target (torch.Tensor)

        Return:
            loss (torch.Tensor)
        """
        identity_weights_norm = F.normalize(self.identity_weights, p=2)
        output = torch.mm(x, identity_weights_norm)

        loss = self.ce_loss(output, target)
        return loss

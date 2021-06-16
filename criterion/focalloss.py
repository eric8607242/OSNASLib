import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, criterion_config):
        """
        Args: criterion_config (dict): The config for criterion
        """
        super(FocalLoss, self).__init__()
        self.criterion_config = criterion_config

        self.alpha = self.criterion_config["alpha"]
        self.gamma = self.criterion_config["gamma"]

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, x, target):
        """
        Args:
            x (torch.Tensor): 
            target (torch.Tensor)

        Return:
            loss (torch.Tensor)
        """
        # Write your code here
        if len(x.shape) == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
            x = x.transpose(1, 2)
            x = x.contiguous().reshape(-1, x.size(2))

        logit = F.log_softmax(x, dim=-1)
        logit = logit.gather(1, target)
        
        pt = logit.exp()

        loss = -self.alpha * ((1 - pt)**self.gamma) * logit
        return loss.mean()

if __name__ == "__main__":
    loss = FocalLoss({"alpha":2, "gamma":2})

    input_x = torch.zeros((10, 10, 32, 32))
    target = torch.zeros((10, 1), dtype=torch.int64)

    print(loss(input_x, target))


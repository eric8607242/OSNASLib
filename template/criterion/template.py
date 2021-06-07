class {{customize_class}}(nn.Module):
    def __init__(self, criterion_config):
        """
        Args: criterion_config (dict): The config for criterion
        """
        super({{customize_class}}, self).__init__()
        self.criterion_config = criterion_config

    def forward(self, x, target):
        """
        Args:
            x (torch.Tensor): 
            target (torch.Tensor)

        Return:
            loss (torch.Tensor)
        """
        # Write your code here
        return loss

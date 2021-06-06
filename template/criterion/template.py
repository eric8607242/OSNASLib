class {{customize_class}}(nn.Module):
    def __init__(self, criterion_config):
        super({{customize_class}}, self).__init__()
        self.criterion_config = criterion_config

    def forward(self, x, target):
        # Write your code here
        return loss

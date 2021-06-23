# How to customize criterion
In OSNASLib, user can easily specific for different tasks by incorporate different components. For supporting various tasks, OSNASLib is able to modify or incorporate customizing loss function easily. In this document, we will briefly introduce how to customize the criterion for various tasks in `./criterion/` easily.

## Generate Interface
```
python3 build_interface.py -t criterion --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]
```

After generating the criterion interface, the file `CUSTOMIZE NAME` will be created in `criterion/`, and the corresponding import will be create automatically.

### Interface Struture
```
- criterion/
    |- [CUSTOMIZE NAME].py
        ...
```

## Criterion Interface
For customizing criterion, the interface should iherited the class `nn.Module`. Besides, to provide more flexibility, you can pass any kind of hyperparameter by the argument `criterion_config`, which is set in the config file.

```python3
# ./criterion/[CUSTOMIZE NAME].py
import torch.nn as nn

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
```

## Setting Config File
After customizing for your criterion, you can utilize your criterion by setting the criterion in the config file easily.

```python3
agent:
    criterion_agent: "[CUSTOMIZE_CLASS]"
```


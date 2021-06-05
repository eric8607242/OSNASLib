# How To Customize Criterion
In OSNASLib, user can easily specific for different tasks by incorporate different components. For supporting various tasks, OSNASLib is able to modify or incorporate customizing loss function easily. In this document, we will briefly introduce how to customize the criterion for various tasks in `./criterion/` easily.

## Generate Template
```
python3 build_template.py -t criterion --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]
```

After generating the criterion template, the file `CUSTOMIZE NAME` will be created in `criterion/`, and the corresponding import will be create automatically.


## Criterion Interface
For customizing criterion, the interface should iherited the class `nn.Module`. Besides, to provide more flexibility, you can pass any kind of hyperparameter by `criterion_config`, which is set in the config file.

```python3
class [CUSTOMIZE_CLASS](nn.Module):
    def __init__(self, criterion_config):
        super(CUSTOMIZE_CLASS, self).__init__()
        self.criterion_config = criterion_config

    def forward(self, x, target):
        # Write your code here
        return loss
```

## Setting Config File
After customing for your criterion, you can utilize your criterion by setting the criterion in the config file easily.

```python3
agent:
    criterion_agent: "[CUSTOMIZE_CLASS]"
```


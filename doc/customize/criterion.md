# How to customize the criterion
In OSNASLib, user can easily specific for different tasks by incorporate different components. For supporting various tasks, OSNASLib is able to modify or incorporate customizing loss function easily. In this document, we will briefly introduce how to customize the criterion for various tasks in `./criterion/` easily.

## Create `customize.py` File or Import Pytorch Loss Function
If the loss function does not implement by Pytorch, you should create a new file for your search strategy.
```
touch ./criterion/customize.py
```

## Criterion Class
For customizing the criterion function.

```
import torch.nn as nn

class CustomizeCriterion(nn.Module):
    def forward():
```

## Import Your Criterion or the Criterion from Pytorch

```python3
from torch.nn import ....
```
or
```python3
from .criterion import CustomizeCriterion
```


## Setting The Criterion Agent In Config File
```python
agent:
    criterion_agent: CustomizeCriterion
```

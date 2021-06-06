# How to customize your training strategy
In OSNASLib, we cover serveral basic training strategy (e.g., differentiable training strategy, uniform training strategy and fairstrictness training strategy) to train the supernet. Besides, we also provide flexible API interface to customize for any your specific tasks or the novel training strategy. In this document, we will briefly introduce how to customize the training strategy in `./training_strategy/` easily.


## Generate Template
```
python3 build_template.py -t training_strategy --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]
```

After generating the training strategy template, the file `[CUSTOMIZE NAME].py` will be created in `training_strategy/` and the corresponding import will be create automatically.


## Training Strategy Interface
For customizing training strategy, the interface class `[CUSTOMIZE CLASS]Sampler` in `[CUSTOMIZE NAME].py` should inherited the base class `BaseSampler`. You shoud implement `step()` method for setting the architecture to train the supernet..


```python3
# ./training_strategy/[CUSTOMIZE NAME].py
import torch
import torch.nn as nn

from .base import BaseSampler

class {{customize_class}}Sampler(BaseSampler):
    def __init__(self, micro_len, macro_len, model):
        super({{customize_class}}, self).__init__(micro_len, macro_len, model)

    def step(self):
        pass
```

## Setting Config File
After customizing for your training strategy, you can utilize your training strategy by setting the training strategy in the config file easily.

```python
agent:
    training_strategy_agent: "[CUSTOMIZE CLASS]Sampler"
```

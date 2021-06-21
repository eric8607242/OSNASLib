# How to customize your training strategy
In OSNASLib, we cover serveral basic training strategy (e.g., differentiable training strategy, uniform training strategy and fairstrictness training strategy) to train the supernet. Besides, we also provide flexible API interface to customize for any your specific tasks or the novel training strategy. In this document, we will briefly introduce how to customize the training strategy in `./training_strategy/` easily.


## Generate Interface
```
python3 build_interface.py -t training_strategy --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]
```

After generating the training strategy interface, the file `[CUSTOMIZE NAME].py` will be created in `training_strategy/` and the corresponding import will be create automatically.

### Interface Struture
```
- training_strategy/
        |- [CUSTOMIZE NAME].py
            ...
```


## Training Strategy Interface
For customizing training strategy, the interface class `[CUSTOMIZE CLASS]Sampler` in `[CUSTOMIZE NAME].py` should inherited the base class `BaseSampler`. You shoud implement `step()` method for setting the architecture to train the supernet.


```python3
# ./training_strategy/[CUSTOMIZE NAME].py
from .base import BaseSampler

class {{customize_class}}Sampler(BaseSampler):
    """ Sampler is the strategy to decide how to train each candidate block of the supernet.
    """
    def __init__(self, micro_len, macro_len, model):
        super({{customize_class}}, self).__init__(micro_len, macro_len, model)

    def step(self):
        """ The sampler step before each iteration

        In each step, the sampler should decide the strategy to update the supernet.
        We provide four protocal in the supernet:
            gumbel_softmax: In each layer, the architecture parameter will be passed into the Gumbel Softmax layer
                            to transform into the weight summation as 1. After weights transformation, the supernet
                            will weighted sum the output of each canididate block. Therefore, user should pass the 
                            hyperparameter `tau` as the argument during `set_forward_state`

            softmax: In each layer, the architecture parameter will be passed into the Softmax layer
                     to transform into the weight summation as 1. After weights transformation, the supernet
                     will weighted sum the output of each canididate block.

            sum:     In each layer, the supernet will weighted sum the output of each candidate block.
                     Therefore, user shoud utilize architecture parameters or set the architecture parameters
                     with supernet.set_arch_param(arch_param)

            single:  In each layer, the supernet will only forward one of all candidate blocks.
                     Therefore, user should set the activate block in each layer 
                     by supernet.set_activate_architecture(architecture)

        User should set the protocal to sum or single by supernet.set_forward_state(state).
        """
        raise NotImplemented
```

## Setting Config File
After customizing for your training strategy, you can utilize your training strategy by setting the training strategy in the config file easily.

```python
agent:
    training_strategy_agent: "[CUSTOMIZE CLASS]Sampler"
```

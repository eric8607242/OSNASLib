# How To Customize Search Space
In OSNASLib, we cover serveral basic search space (e.g., FBNet, ProxylessNAS, and SPOS). We split the config of the search space into micro config and macro config. By designing different micro config and macro config, user can design the different search space easily. In this document, we will briefly introduce how to customize the search space in `./model/` easily.

## Generate Template
```
python3 build_template.py -t model --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]
```

After generating the model template, the directory `[CUSTOMIZE NAME]/` will be created in `model/`, and the corresponding files (`__init__.py` and `[CUSTOMIZE NAME]_supernet.py`) are created in the directory `[CUSTOMIZE NAME]/`.

### Template Struture
```
- model/
    |- [CUSTOMIZE NAME]/
    |         |- __init__.py
    |         |- [CUSTOMIZE NAME]_supernet.py
            ...
```


## Model Interface
For customizing model, the interface class `[CUSTOMIZE CLASS]Supernet` in `[CUSTOMIZE NAME]_supernet.py` should inherit the class `BaseSupernet` and implement the staticmethod `get_model_cfg` to return `macro_cfg` and `micro_cfg`.

```python3
# ./model/[CUSTOMIZE NAME]/[CUSTOMIZE NAME]_supernet.py
from ..base import BaseSupernet

class {{customize_class}}(BaseSupernet):
    @staticmethod
    def get_model_cfg(classes):
        """ Return the macro and micro configurations of the search space.

        Args:
            classes (int): The number of output class (dimension).
        
        Return:
            macro_cfg (dict): The structure of the entire supernet. The structure is split into three parts, "first", "search", "last"
            micro_cfg (list): The all configurations in each layer of supernet.
        """
        # block_type, kernel_size, se, activation, kwargs
        micro_cfg = []

        macro_cfg = {
                # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
                "first": [],
                # in_channels, out_channels, stride
                "search": [],
                # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
                "last": []}
        return macro_cfg, micro_cfg

```

### Type Of Candidate Block
In OSNASLib, we provide serveral type of candidate block as follows:
1. MobileNet Block (Inverted residual bottleneck)
2. ShuffleNet Block
3. ShuffleNetX Block
4. Linear (Classifier)
5. Global Average Pooling
6. Conv BN Activation Block
7. Skip Connection Block

You can crate new candidate block in `./model/block_builder.py` by following candidate block interface:
```
def _get_[CUSTOMIZE NAME]_block(in_channels, out_channels, kernel_size,
        stride, activation, se, bn_momentum, bn_track_running_stats, *args, **kwargs):

    return block
```



## Setting Config File
After customizing for your search space, you can utilize your search space by setting the search space into the config file easily.
```
agent:
    supernet_agent: "[CUSTOMIZE CLASS]Supernet"
```

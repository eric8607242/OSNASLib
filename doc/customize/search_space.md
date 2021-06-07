# How To Customize Search Space
In OSNASLib, we cover serveral basic search space (e.g., FBNet, ProxylessNAS, and SPOS). We split the config of the search space into micro config and macro config. By designing different micro config and macro config, user can design the different search space easily. In this document, we will briefly introduce how to customize the search space in `./model/` easily.

## Generate Template
```
python3 build_template.py -t model --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]
```

After generating the model template, the directory `[CUSTOMIZE NAME]/` will be created in `model/`, and the corresponding files (`__init__.py` and `[CUSTOMIZE NAME]_supernet.py`) are created in the directory `[CUSTOMIZE NAME]/`.


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

## Setting Config File
After customizing for your search space, you can utilize your search space by setting the search space into the config file easily.
```
agent:
    supernet_agent: "[CUSTOMIZE CLASS]Supernet"
```

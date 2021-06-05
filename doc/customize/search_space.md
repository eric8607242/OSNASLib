# How To Customize Search Space
In OSNASLib, we cover serveral basic search space (e.g., FBNet, ProxylessNAS, and SPOS). We split the config of the search space into micro config and macro config. By designing different micro config and macro config, user can design the different search space easily. In this document, we will briefly introduce how to customize the search space in `./model/` easily.

## Generate Template
```
python3 build_template.py -t model --customize-name [CUSTOMIZE NAME] --customize-class [CUSTOMIZE CLASS]
```

After generating the model template, the directory `CUSTOMIZE NAME` will be created in `model/`, and the corresponding files (`__init__.py` and `[CUSTOMIZE NAME]_supernet.py`) are created in the directory `CUSTOMIZE NAME`.


## Model Interface
For customizing model, the interface in `[CUSTOMIZE NAME]_supernet.py` should inherit the class `BaseSupernet` and implement the staticmethod `get_model_cfg` to return `macro_cfg` and `micro_cfg`.

```python3
from ..base import BaseSupernet

class [CUSTOMIZE_CLASS]Supernet(BaseSupernet):
    @staticmethod
    def get_model_cfg(classes):
        return macro_cfg, micro_cfg
```

## Setting Config File
After customizing for your search space, you can utilize your search space by setting the search space into the config file easily.
```
agent:
    supernet_agent: "[CUSTOMIZE_CLASS]Supernet"
```

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
For customizing model, the interface class `[CUSTOMIZE CLASS]Supernet` and `[CUSTOMIZE CLASS]Superlayer` in `[CUSTOMIZE NAME]_supernet.py` should inherit the class `BaseSupernet` and `BaseSuperlayer`, respectively. The staticmethod `get_model_cfg` return `macro_cfg` and `micro_cfg`.

There are lots of method should be implemented by user, which aims at providing more flexibility for each kind of supernet structure and allowing each search strategy can search on each search space. For example, the method `set_activate_architecture()` is utilized by single-path NAS and the method `initialize_arch_param()` is utilized by differentiable NAS. If you only want to search by the specific search strategy, you can just implement the corresponding methods.


```python3
# ./model/[CUSTOMIZE NAME]/[CUSTOMIZE NAME]_supernet.py
from ..base import BaseSupernet, BaseSuperlayer

class [CUSTOMIZE CLASS]Superlayer(BaseSuperlayer):
    def _construct_supernet_layer(self, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats):
        supernet_layer = nn.ModuleList()
        for b_cfg in micro_cfg:
            block_type, kernel_size, se, activation, kwargs = b_cfg
            block = get_block(block_type=block_type,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation=activation,
                              se=se,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **kwargs
                              )
            supernet_layer.append(block)
        return supernet_layer

    def forward(self, x):
        return x

    # Single-path NAS
    def set_activate_architecture(self, architecture):
        """ Activate the path based on the architecture. Utilizing in single-path NAS.

        Args:
            architecture (torch.tensor): The block index for each layer.
        """
        pass

    # Differentaible NAS
    def set_arch_param(self, arch_param):
        """ Set architecture parameter directly

        Args:
            arch_param (torch.tensor)
        """
        pass
        
    def initialize_arch_param(self):
        pass

    def get_arch_param(self):
        """ Return architecture parameters.

        Return:
            self.arch_param (nn.Parameter)
        """
        pass

    def get_best_arch_param(self):
        """ Get the best neural architecture from architecture parameters (argmax).

        Return:
            best_architecture (np.ndarray)
        """
        pass

    def set_forward_state(self, state):
        """ Set supernet forward state. ["single", "sum"]

        Args:
            state (str): The state in model forward.
        """
        pass



class [CUSTOMIZE CLASS]Supernet(BaseSupernet):
    superlayer_builder = [CUSTOMIZE CLASS]Superlayer

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

    def get_model_cfg_shape(self):
        """ Return the shape of model config for the architecture generator.

        Return 
            model_cfg_shape (Tuple)
        """
        return model_cfg_shape

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

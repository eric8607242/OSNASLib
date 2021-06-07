from ..base import BaseSupernet

class ProxylessNASSupernet(BaseSupernet):
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
        micro_cfg =  [["mobile", 3, False, "relu", {"expansion_rate": 3}],
                          ["mobile", 3, False, "relu", {"expansion_rate": 6}],
                          ["mobile", 5, False, "relu", {"expansion_rate": 3}],
                          ["mobile", 5, False, "relu", {"expansion_rate": 6}],
                          ["mobile", 7, False, "relu", {"expansion_rate": 3}],
                          ["mobile", 7, False, "relu", {"expansion_rate": 6}],
                          ["skip", 0, False, "relu", {}]]
        macro_cfg = {
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "first": [["conv", 3, 32, 2, 3, "relu", False, {}],  # stride 1 for CIFAR
                      ["mobile", 32, 16, 1, 3, "relu", False, {"expansion_rate": 1}]],
            # in_channels, out_channels, stride
            "search": [[16, 24, 2],  # stride 1 for CIFAR
                       [24, 24, 1],
                       [24, 24, 1],
                       [24, 24, 1],
                       [24, 40, 2],
                       [40, 40, 1],
                       [40, 40, 1],
                       [40, 40, 1],
                       [40, 80, 2],
                       [80, 80, 1],
                       [80, 80, 1],
                       [80, 80, 1],
                       [80, 96, 1],
                       [96, 96, 1],
                       [96, 96, 1],
                       [96, 96, 1],
                       [96, 192, 2],
                       [192, 192, 1],
                       [192, 192, 1],
                       [192, 192, 1],
                       [192, 320, 1]],
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "last": [["conv", 320, 1280, 1, 1, "relu", False, {}],
                     ["global_average", 0, 0, 0, 0, 0, 0, {}],
                     ["classifier", 1280, classes, 0, 0, 0, 0, {}]]
        }
        return macro_cfg, micro_cfg

from ..base import BaseSupernet

class SPOSSupernet(BaseSupernet):
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
        micro_cfg = [["shuffle", 3, False, "relu", {}],
                          ["shuffle", 5, False, "relu", {}],
                          ["shuffle", 7, False, "relu", {}],
                          ["shuffleX", 0, False, "relu", {}]]
        macro_cfg = {
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "first": [["Conv", 3, 16, 2, 3, "relu", False, {}]],  # stride 1 for CIFAR
            # in_channels, out_channels, stride
            "search": [[16, 64, 2],  # stride 1 for CIFAR
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 160, 2],
                       [160, 160, 1],
                       [160, 160, 1],
                       [160, 160, 1],
                       [160, 320, 2],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 320, 1],
                       [320, 640, 2],
                       [640, 640, 1],
                       [640, 640, 1],
                       [640, 640, 1]],
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "last": [["conv", 640, 1024, 1, 1, "relu", False, {}],
                     ["global_average", 0, 0, 0, 0, 0, 0, {}],
                     ["classifier", 1024, classes, 0, 0, 0, 0, {}]]
        }
        return macro_cfg, micro_cfg

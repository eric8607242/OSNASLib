from ..base import BaseSupernet

class MobileFaceNetSupernet(BaseSupernet):
    @staticmethod
    def get_model_cfg(classes):
        micro_cfg = [["mobile", 3, False, "prelu", {"expansion_rate": 1}],
                           ["mobile", 3, False, "prelu", {"expansion_rate": 2}],
                           ["mobile", 3, False, "prelu", {"expansion_rate": 4}],
                           ["mobile", 5, False, "prelu", {"expansion_rate": 1}],
                           ["mobile", 5, False, "prelu", {"expansion_rate": 2}],
                           ["mobile", 5, False, "prelu", {"expansion_rate": 4}],
                           ["skip", 0, False, "prelu", {}]]
        macro_cfg = {
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "first": [["conv", 3, 64, 2, 3, "prelu", False, {}],
                ["conv", 64, 64, 1, 3, "prelu", False, {"group":64}]],  # stride 1 for CIFAR
            # in_channels, out_channels, stride
            "search": [[64, 64, 2],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 64, 1],
                       [64, 128, 2],
                       [128, 128, 1],
                       [128, 128, 1],
                       [128, 128, 1],
                       [128, 128, 1],
                       [128, 128, 1],
                       [128, 128, 1],
                       [128, 128, 2],
                       [128, 128, 1],
                       [128, 128, 1]],
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "last": [
                     ["conv", 128, 512, 1, 1, "prelu", False, {}],
                     ["conv", 512, 512, 1, 7, None, False, {"group":512}],
                     ["conv", 512, classes, 1, 1, None, False, {}],
                     ["global_average", 0, 0, 0, 0, 0, 0, {}]]
        }
        return macro_cfg, micro_cfg

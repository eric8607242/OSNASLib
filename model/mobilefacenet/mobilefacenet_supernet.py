from ..base import BaseSupernet

class MobileFaceNetSupernet(BaseSupernet):
    def _init_model_config(self, classes):
        self.micro_cfg = [["Mobile", 3, False, "prelu", {"expansion_rate": 2}],
                           ["Mobile", 3, False, "prelu", {"expansion_rate": 4}],
                           ["Mobile", 3, False, "prelu", {"expansion_rate": 6}],
                           ["Mobile", 5, False, "prelu", {"expansion_rate": 2}],
                           ["Mobile", 5, False, "prelu", {"expansion_rate": 4}],
                           ["Mobile", 5, False, "prelu", {"expansion_rate": 6}],
                           ["Skip", 0, False, "prelu", {}]]
        self.macro_cfg = {
            # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
            "first": [["Conv", 3, 64, 2, 3, "prelu", False, {}],
                ["Conv", 64, 64, 1, 3, "prelu", False, {"group":64}]],  # stride 1 for CIFAR
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
                     ["Conv", 128, 512, 1, 1, "prelu", False, {}],
                     ["Conv", 512, 512, 1, 7, None, False, {"group":512}],
                     ["Conv", 512, classes, 1, 1, None, False, {}],
                     ["global_average", 0, 0, 0, 0, 0, 0, {}]]
        }


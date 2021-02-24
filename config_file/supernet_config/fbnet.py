# block_type, kernel_size, se, activation, kwargs
FBNET_MICRO_CFG = [["Mobile", 3, False, "relu", {"expansion_rate": 1, "point_group": 1}],
                   ["Mobile", 3, False, "relu", {"expansion_rate": 1, "point_group": 2}],
                   ["Mobile", 3, False, "relu", {"expansion_rate": 3, "point_group": 1}],
                   ["Mobile", 3, False, "relu", {"expansion_rate": 6, "point_group": 1}],
                   ["Mobile", 5, False, "relu", {"expansion_rate": 1, "point_group": 1}],
                   ["Mobile", 5, False, "relu", {"expansion_rate": 1, "point_group": 2}],
                   ["Mobile", 5, False, "relu", {"expansion_rate": 3, "point_group": 1}],
                   ["Mobile", 5, False, "relu", {"expansion_rate": 6, "point_group": 1}],
                   ["Skip", 0, False, "relu", {}]]


FBNET_SUPERNET_S_CFG = {
    # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
    "first": [["Conv", 3, 16, 2, 3, "relu", False, {}]],  # stride 1 for CIFAR
    # in_channels, out_channels, stride
    "search": [[16, 16, 1],
               [16, 24, 2],  # stride 1 for CIFAR
               [24, 24, 1],
               [24, 24, 1],
               [24, 24, 1],
               [24, 32, 2],
               [32, 32, 1],
               [32, 32, 1],
               [32, 32, 1],
               [32, 64, 2],
               [64, 64, 1],
               [64, 64, 1],
               [64, 64, 1],
               [64, 112, 1],
               [112, 112, 1],
               [112, 112, 1],
               [112, 112, 1],
               [112, 184, 2],
               [184, 184, 1],
               [184, 184, 1],
               [184, 184, 1],
               [184, 352, 1]],
    # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
    "last": [["Conv", 352, 1504, 1, 1, "relu", False, {}],
             ["global_average", 0, 0, 0, 0, 0, 0, {}],
             ["classifier", 1504, 1000, 0, 0, 0, 0, {}]]
}

FBNET_SUPERNET_L_CFG = {
    # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
    "first": [["Conv", 3, 16, 2, 3, "relu", False, {}]],  # stride 1 for CIFAR
    # in_channels, out_channels, stride
    "search": [[16, 16, 1],
               [16, 24, 2],  # stride 1 for CIFAR
               [24, 24, 1],
               [24, 24, 1],
               [24, 24, 1],
               [24, 32, 2],
               [32, 32, 1],
               [32, 32, 1],
               [32, 32, 1],
               [32, 64, 2],
               [64, 64, 1],
               [64, 64, 1],
               [64, 64, 1],
               [64, 112, 1],
               [112, 112, 1],
               [112, 112, 1],
               [112, 112, 1],
               [112, 184, 2],
               [184, 184, 1],
               [184, 184, 1],
               [184, 184, 1],
               [184, 352, 1]],
    # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
    "last": [["Conv", 352, 1984, 1, 1, "relu", False, {}],
             ["global_average", 0, 0, 0, 0, 0, 0, {}],
             ["classifier", 1984, 1000, 0, 0, 0, 0, {}]]
}

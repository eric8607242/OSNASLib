# block_type, kernel_size, se, activation, kwargs
SPOS_MICRO_CFG = [["Shuffle", 3, False, "relu", {}],
                  ["Shuffle", 5, False, "relu", {}],
                  ["Shuffle", 7, False, "relu", {}],
                  ["ShuffleX", 0, False, "relu", {}]]



SPOS_SUPERNET_CFG = {
        # block_type, in_channels, out_channels, stride, kernel_size, activation, se, kwargs
        "first":[["Conv", 3, 16, 2, 3, "relu", False, {}]], # stride 1 for CIFAR
        # in_channels, out_channels, stride
        "search":[[16, 64, 2], # stride 1 for CIFAR
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
        "last":[["Conv", 640, 1024, 1, 1, "relu", False, {}],
                ["global_average", 0, 0, 0, 0, 0, 0, {}],
                ["classifier", 1024, 1000, 0, 0, 0, 0, {}]]
        }

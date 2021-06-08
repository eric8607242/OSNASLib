from ..base import BaseSupernet, BaseSuperlayer

class SPOSSuperlayer(BaseSuperlayer):
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
        if self.forward_state == "gumbel_sum":
            weight = F.gumbel_softmax(self.arch_param[i], dim=0) 
            x = sum(p * b(x) for p, b in zip(weight, l))

        elif self.forward_state == "sum":
            weight = F.softmax(self.arch_param[i], dim=0)
            x = sum(p * b(x) for p, b in zip(weight, l))

        elif self.forward_state == "single":
            x = l[self.architecture[i]](x)
        return x

    def set_activate_architecture(self, architecture):
        """ Activate the path based on the architecture. Utilizing in single-path NAS.

        Args:
            architecture (torch.tensor): The block index for each layer.
        """
        self.architecture = architecture

    # Differentaible NAS
    def set_arch_param(self, arch_param):
        """ Set architecture parameter directly

        Args:
            arch_param (torch.tensor)
        """
        self.arch_param = arch_param
        

    def initialize_arch_param(self):
        micro_len = len(self.micro_cfg)

        self.arch_param = nn.Parameter(
            1e-3 * torch.randn((1, micro_len), requires_grad=False))


    def get_arch_param(self):
        """ Return architecture parameters.

        Return:
            self.arch_param (nn.Parameter)
        """
        return self.arch_param


    def get_best_arch_param(self):
        """ Get the best neural architecture from architecture parameters (argmax).

        Return:
            best_architecture (np.ndarray)
        """
        best_architecture = self.arch_param.data.argmax(dim=1)
        best_architecture = best_architecture.cpu().numpy()

        return best_architecture


    def set_forward_state(self, state):
        """ Set supernet forward state. ["single", "sum"]

        Args:
            state (str): The state in model forward.
        """
        self.forward_state = state

        
class SPOSSupernet(BaseSupernet):
    superlayer_builder = SPOSSuperlayer

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

    def get_model_cfg_shape(self):
        """ Return the shape of model config for the architecture generator.

        Return 
            model_cfg_shape (Tuple)
        """
        return (len(macro_cfg["search"]), len(micro_cfg))

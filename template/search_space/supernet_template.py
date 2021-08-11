from ..base_supernet import BaseSupernet, BaseSuperlayer

class {{customize_class}}Superlayer(BaseSuperlayer):
    def _construct_supernet_layer(self, in_channels, out_channels, stride, bn_momentum, bn_track_running_stats, *args, **kwargs):
        """ Construct the supernet layer module.
        """
        self.supernet_layer = nn.ModuleList()
        for b_cfg in self.micro_cfg:
            block_type, kernel_size, se, activation, cfg_kwargs = b_cfg
            block = get_block(block_type=block_type,
                              in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              activation=activation,
                              se=se,
                              bn_momentum=bn_momentum,
                              bn_track_running_stats=bn_track_running_stats,
                              **cfg_kwargs
                              )
            self.supernet_layer.append(block)


class {{customize_class}}Supernet(BaseSupernet):
    superlayer_builder = {{customize_class}}Superlayer

    @staticmethod
    def get_search_space_cfg(classes):
        """ Return the macro and micro configurations of the search space.

        Args:
            classes (int): The number of output class (dimension).
        
        Return:
            search_space_cfg (list): list of search space tuple, there are macro_cfg and micro_cfg in each tuple.
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
        search_space_cfg = [(macro_cfg, micro_cfg)]
        return search_space_cfg

    def get_seach_space_cfg_shape(self):
        """ Return the shape of search space config for the architecture generator.

        Return 
            search_space_cfg_shape (list)
        """
        search_space_cfg_shape = [(len(macro_cfg), len(micro_cfg)) for macro_cfg, micro_cfg in self.search_space_cfg]
        return search_space_cfg_shape

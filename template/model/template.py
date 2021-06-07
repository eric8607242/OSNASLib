from ..base import BaseSupernet

class {{customize_class}}(BaseSupernet):
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


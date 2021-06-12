from ..base_lookup_table import LookUpTable
from ..block_builder import get_block

class ProxylessNASLookUpTable(LookUpTable):
    def construct_info_table(self, info_metric_list=["flops", "param", "latency"]):
        """ Construct the info lookup table of search stage in macro config.

        We provide serveral useful method to calculate the info metric and process info
        metric table. Please refer to `/model/base_lookup_table.py` for more details.

        Args:
            info_metric_list (list):

        Return:
            info_table (dict)
        """
        input_size = self.get_search_input_size()
        info_table = {metric: [] for metric in info_metric_list}

        for l, l_cfg in enumerate(self.macro_cfg["search"]):
            in_channels, out_channels, stride = l_cfg
            layer_info = {metric: [] for metric in info_metric_list}

            for b, b_cfg in enumerate(self.micro_cfg):
                block_type, kernel_size, se, activation, kwargs = b_cfg
                block = get_block(block_type=block_type,
                                  in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  activation=activation,
                                  se=se,
                                  bn_momentum=0.1,
                                  bn_track_running_stats=True,
                                  **kwargs
                                  )

                block_info = self._get_block_info(
                    block, in_channels, input_size, info_metric_list)
                layer_info = self._merge_info_table(
                    layer_info, block_info, info_metric_list)

            input_size = input_size if stride == 1 else input_size // 2
            info_table = self._merge_info_table(
                info_table, layer_info, info_metric_list)


        return info_table



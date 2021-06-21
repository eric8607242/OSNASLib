import math

from ..base_lookup_table import LookUpTable
from ..block_builder import get_block

class SGNASLookUpTable(LookUpTable):
    def construct_info_table(self, info_metric_list=["flops", "param", "latency"]):
        """ Construct the info lookup table of search stage in macro config.

        We provide serveral useful method to calculate the info metric and process info
        metric table. Please refer to `/model/base_lookup_table.py` for more details.

        Args:
            info_metric_list (list):

        Return:
            info_table (dict)
        """
        max_expansion_rate = self.micro_cfg[0][-1]["max_expansion_rate"]
        min_expansion_rate = self.micro_cfg[0][-1]["min_expansion_rate"]

        input_size = self.get_search_input_size()
        info_table = {metric: [] for metric in info_metric_list}
        for l, l_cfg in enumerate(self.macro_cfg["search"]):
            in_channels, out_channels, stride = l_cfg

            hidden_channel = int(in_channels * max_expansion_rate)

            # Calculate pointwise info
            block = get_block(block_type="conv",
                              in_channels=in_channels,
                              out_channels=hidden_channel,
                              kernel_size=1,
                              stride=1,
                              activation="relu",
                              se=False,
                              bn_momentum=0.1,
                              bn_track_running_stats=True)
            pointwise_info = self._get_block_info(block, in_channels, input_size, info_metric_list)

            # Calculate pointwise_1 info
            block = get_block(block_type="conv",
                              in_channels=hidden_channel,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              activation=None,
                              se=False,
                              bn_momentum=0.1,
                              bn_track_running_stats=True)
            
            pointwise_1_info = self._get_block_info(block, hidden_channel, input_size if stride == 1 else input_size // 2, info_metric_list)
            block_in_channels = hidden_channel // max_expansion_rate
            block_out_channels = hidden_channel // max_expansion_rate

            for e in range(max_expansion_rate):
                # Calculate unified sub block info
                layer_info = {metric: [] for metric in info_metric_list}
                for b, b_cfg in enumerate(self.micro_cfg):
                    block_type, kernel_size, se, activation, kwargs = b_cfg
                    if kernel_size == 0:
                        block_info = {}
                        for metric in info_metric_list:
                            block_info[metric] = 0                    
                    else:
                        block = get_block(block_type=block_type,
                                          in_channels=block_in_channels,
                                          out_channels=block_out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          activation=activation,
                                          se=se,
                                          bn_momentum=0.1,
                                          bn_track_running_stats=True,
                                          **kwargs)
                        
                        block_info = self._get_block_info(
                            block, block_in_channels, input_size, info_metric_list)

                        # Count pointwise info
                        for metric in info_metric_list:
                            block_info[metric] += (pointwise_info[metric] + pointwise_1_info[metric]) / max_expansion_rate

                    layer_info = self._merge_info_table(
                        layer_info, block_info, info_metric_list)
                info_table = self._merge_info_table(
                    info_table, layer_info, info_metric_list)

            input_size = input_size if stride == 1 else math.ceil(input_size/2)


        return info_table



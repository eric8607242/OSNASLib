from ..base import BaseSupernet

from .fbnet_config import FBNET_SUPERNET_S_CFG, FBNET_SUPERNET_L_CFG, FBNET_MICRO_CFG

class FBNetSSupernet(BaseSupernet):
    macro_cfg = FBNET_SUPERNET_S_CFG
    micro_cfg = FBNET_MICRO_CFG


class FBNetLSupernet(BaseSupernet):
    macro_cfg = FBNET_SUPERNET_L_CFG
    micro_cfg = FBNET_MICRO_CFG


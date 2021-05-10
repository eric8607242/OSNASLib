from ..base import BaseSupernet

from .spos_config import SPOS_SUPERNET_CFG, SPOS_MICRO_CFG

class SPOSSupernet(BaseSupernet):
    macro_cfg = SPOS_SUPERNET_CFG
    micro_cfg = SPOS_MICRO_CFG

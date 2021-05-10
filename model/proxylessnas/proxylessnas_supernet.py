from ..base import BaseSupernet

from .proxylessnas_config import PROXYLESSNAS_SUPERNET_S_CFG, PROXYLESSNAS_MICRO_CFG

class ProxylessNASSupernet(BaseSupernet):
    macro_cfg = PROXYLESSNAS_SUPERNET_S_CFG
    micro_cfg = PROXYLESSNAS_MICRO_CFG

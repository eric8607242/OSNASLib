from .fbnet import FBNET_SUPERNET_S_CFG, FBNET_SUPERNET_L_CFG, FBNET_MICRO_CFG
from .proxylessnas import PROXYLESSNAS_SUPERNET_S_CFG, PROXYLESSNAS_MICRO_CFG
from .spos import SPOS_SUPERNET_CFG, SPOS_MICRO_CFG

def get_supernet_cfg(search_space, classes):
    if search_space == "fbnet_s":
        supernet_cfg = FBNET_SUPERNET_S_CFG
        micro_cfg = FBNET_MICRO_CFG

    elif search_space == "fbnet_l":
        supernet_cfg = FBNET_SUPERNET_L_CFG
        micro_cfg = FBNET_MICRO_CFG

    elif search_space == "proxylessnas":
        supernet_cfg = PROXYLESSNAS_SUPERNET_S_CFG
        micro_cfg = PROXYLESSNAS_MICRO_CFG

    elif search_space == "spos":
        supernet_cfg = SPOS_SUPERNET_CFG
        micro_cfg = SPOS_MICRO_CFG

    else:
        raise NotImplementedError


    # Revise the output channel of classifier
    supernet_cfg["last"][-1][2] = classes

    return supernet_cfg, micro_cfg



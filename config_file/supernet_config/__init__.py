from .fbnet import FBNET_SUPERNET_S_CFG, FBNET_SUPERNET_L_CFG, FBNET_MICRO_CFG
from .proxylessnas import PROXYLESSNAS_SUPERNET_S_CFG, PROXYLESSNAS_MICRO_CFG
from .spos import SPOS_SUPERNET_CFG, SPOS_MICRO_CFG

def get_supernet_cfg(search_space, classes, dataset):
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

    # Modify the stride for cifar or imagenet
    if dataset == "cifar100" or dataset == "cifar10":
        supernet_cfg["first"][0][3] = 1
        for l in range(len(supernet_cfg["search"])):
            if supernet_cfg["search"][l][2] == 2:
                supernet_cfg["search"][l][2] = 1
                break


    # Modify the output channel of classifier
    supernet_cfg["last"][-1][2] = classes

    return supernet_cfg, micro_cfg



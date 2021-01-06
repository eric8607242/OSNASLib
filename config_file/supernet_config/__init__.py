from .fb_net import FB_SUPERNET_S_CFG, FB_SUPERNET_L_CFG, FB_MICRO_CFG

def get_supernet_cfg(search_space, classes):
    if search_space == "fbnet_s":
        supernet_cfg = FB_SUPERNET_S_CFG
        micro_cfg = FB_MICRO_CFG
    elif search_space == "fbnet_l":
        supernet_cfg = FB_SUPERNET_L_CFG
        micro_cfg = FB_MICRO_CFG
    else:
        raise NotImplementedError


    # Revise the output channel of classifier
    supernet_cfg["last"][-1][2] = classes

    return supernet_cfg, micro_cfg



import numpy as np

import torch
import torch.nn as nn

def random_search(trainer, training_strategy, supernet, val_loader, lookup_table, target_hc, logger, random_iteration=1000, info_metric="flops"):

    random_architectures = []
    for i in range(random_iteration):
        logger.info("Architecture index : {}".format(i))
        
        architecture = training_strategy.generate_training_architecture()
        architecture_info = lookup_table.get_model_info(architecture, info_metric=info_metric)

        while architecture_info > target_hc:
            architecture = training_strategy.generate_training_architecture()
            architecture_info = lookup_table.get_model_info(architecture, info_metric=info_metric)
        random_architectures.append(architecture)
    
    architectures_top1_acc = []
    for a in random_architectures:
        supernet.module.set_activate_architecture(architecture) if isinstance(supernet, nn.DataParallel) else supernet.set_activate_architecture(architecture)
        architectures_top1_acc.append(trainer.validate(supernet, val_loader))


    architectures_top1_acc = np.array(architectures_top1_acc)
    max_top1_acc_index = architectures_top1_acc.argmax()
    logger.info("Random search maximum top1 acc : {}".format(architectures_top1_acc[max_top1_acc_index]))

    return random_architectures[max_top1_acc_index]

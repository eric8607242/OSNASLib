import os
import time

import torch
import torch.nn as nn

from config_file.arg_config import *
from config_file.supernet_config import *

from lib import *
from search_strategy import *

if __name__ == "__main__":
    args = get_init_config("search")

    logger = get_logger(args.logger_path)
    writer = get_writer(args.title, args.random_seed, args.writer_path)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)
    
    device = torch.device(args.device)

    macro_cfg, micro_cfg = get_supernet_cfg(args.search_space, args.classes, args.dataset)
    supernet = Supernet(macro_cfg, micro_cfg, args.classes, 
                        args.dataset, args.search_strategy, 
                        bn_momentum=args.bn_momentum, 
                        bn_track_running_stats=args.bn_track_running_stats)

    supernet = supernet.to(device)

    train_loader, val_loader = get_train_loader(args.dataset, args.dataset_path, args.batch_size, args.num_workers, train_portion=args.train_portion)

    training_strategy = TrainingStrategy(args.sample_strategy, len(micro_cfg), len(macro_cfg["search"]), supernet)

    lookup_table = LookUpTable(macro_cfg, micro_cfg, args.lookup_table_path, args.input_size, info_metric=["flops", "param"])

    optimizer = get_optimizer(supernet.parameters(), args.optimizer, 
                                  learning_rate=args.lr, 
                                  weight_decay=args.weight_decay, 
                                  logger=logger,
                                  momentum=args.momentum, 
                                  alpha=args.alpha, 
                                  beta=args.beta)

    lr_scheduler = get_lr_scheduler(optimizer, args.lr_scheduler, logger, 
                                    step_per_epoch=len(train_loader), 
                                    step_size=args.decay_step, 
                                    decay_ratio=args.decay_ratio,
                                    total_epochs=args.epochs)

    criterion = get_criterion()

    start_epoch = 0
    if args.resume:
        start_epoch = resume_checkpoint(supernet, args.resume, optimizer, lr_scheduler)
        logger.info("Resume training from {} at epoch {}".format(args.resume, start_epoch))

    if device.type == "cuda" and args.ngpu >= 1:
        supernet = nn.DataParallel(supernet, list(range(args.ngpu)))

    search_strategy = SearchStrategy(supernet, val_loader, lookup_table, args.search_strategy, args, logger, device)

    trainer = Trainer(criterion, optimizer, lr_scheduler, writer, logger, args.device, "search", args, training_strategy=training_strategy, search_strategy=search_strategy, start_epoch=start_epoch)
    start_time = time.time()

    if not args.directly_search:
        trainer.train_loop(supernet, train_loader, val_loader)

    best_architecture, best_architecture_hc, best_architecture_top1 = search_strategy.search(trainer, training_strategy)
    logger.info("Best architectrue : {}".format(best_architecture))
    logger.info("Best architectrue top1 : {:.3f}".format(best_architecture_top1*100))
    logger.info("Best architectrue hc : {}".format(best_architecture_hc))

    save_architecture(args.searched_model_path, best_architecture)
    logger.info("Total search time : {:.2f}".format(time.time() - start_time))

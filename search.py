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

    macro_cfg, micro_cfg = get_supernet_cfg(args.search_space, args.classes)
    supernet = Supernet(macro_cfg, micro_cfg, args.classes, args.dataset, args.sample_strategy)

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

    if args.resume:
        optimizer_state, resume_epoch = resume_checkpoint(model, args.resume)
        optimizer.load_state_dict(optimizer_state["optimizer"])
        start_epoch = resume_epoch

    if device.type == "cuda" and args.ngpu >= 1:
        supernet = supernet.to(device)
        supernet = nn.DataParallel(supernet, list(range(args.ngpu)))

    search_strategy = SearchStrategy(supernet, val_loader, args.search_strategy, args, logger)

    trainer = Trainer(criterion, optimizer, args.epochs, writer, logger, args.device, "search", training_strategy=training_strategy, search_strategy=search_strategy)
    trainer.train_loop(supernet, train_loader, val_loader)

    best_architecture = search_strategy.search(trainer, training_strategy, val_loader, lookup_table)

    save_architecture(args.searched_model_path, best_architecture)
    logger.info("Total search time : {:.2f}".format(time.time() - start_time))

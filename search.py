from config_file.arg_config import *
from config_file.supernet_config import *

from lib import *
from search_strategy import *

if __name__ == "__main__":
    args = get_init_config()

    macro_cfg, micro_cfg = get_supernet_cfg(args.search_space, args.classes)
    supernet = Supernet(macro_cfg, micro_cfg, args.classes, args.dataset, args.sample_strategy)

    train_loader, val_loader = get_train_loader(args.dataset, args.dataset_path, args.batch_size, args.num_workers, train_portion=args.train_portion)

    training_strategy = TrainingStrategy(args.sample_strategy, len(micro_cfg), len(macro_cfg["search"]))

    lookup_table = LookUpTable(macro_cfg, micro_cfg, args.lookup_table_path, args.input_size, info_metric=["flops", "param"])


    logger = get_logger(args.logger_path)
    writer = get_writer(args.title, args.random_seed, args.writer_path)


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

    trainer = Trainer(criterion, optimizer, args.epochs, writer, logger, args.device, training_strategy)

    #trainer.train_loop(supernet, train_loader, val_loader)

    search_strategy = SearchStrategy(supernet, args.search_strategy, args, logger)
    best_architecture = search_strategy.search(trainer, training_strategy, val_loader, lookup_table)


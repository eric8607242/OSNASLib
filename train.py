# Train the architecture searched by NAS algorithm
import time

import torch
import torch.nn as nn

from config_file.arg_config import *
from config_file.supernet_config import *

from lib import *
from search_strategy import *

if __name__ == "__main__":
    args = get_init_config("evaluate")

    logger = get_logger(args.logger_path)
    writer = get_writer(args.title, args.random_seed, args.writer_path)

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    device = torch.device(args.device)

    architecture = load_architecture(args.searched_model_path)

    macro_cfg, micro_cfg = get_supernet_cfg(
        args.search_space, args.classes, args.dataset)
    model = Model(
        macro_cfg,
        micro_cfg,
        architecture,
        args.classes,
        args.dataset)
    model = model.to(device)

    train_loader, val_loader = get_train_loader(
        args.dataset, args.dataset_path, args.batch_size, args.num_workers, train_portion=args.train_portion)
    test_loader = get_test_loader(
        args.dataset,
        args.dataset_path,
        args.batch_size,
        args.num_workers)

    optimizer = get_optimizer(model.parameters(), args.optimizer,
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
        start_epoch = resume_checkpoint(
            model, args.resume, optimizer, lr_scheduler)
        logger.info(
            "Resume training from {} at epoch {}".format(
                args.resume, start_epoch))

    if device.type == "cuda" and args.ngpu >= 1:
        model = nn.DataParallel(model, list(range(args.ngpu)))

    trainer = Trainer(
        criterion,
        optimizer,
        lr_scheduler,
        writer,
        logger,
        args.device,
        "evaluate",
        args,
        start_epoch=start_epoch)
    start_time = time.time()

    trainer.train_loop(model, train_loader, test_loader)
    logger.info("Total search time : {:.2f}".format(time.time() - start_time))

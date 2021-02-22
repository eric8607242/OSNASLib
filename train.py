# Train the architecture searched by NAS algorithm
import time

from config_file.arg_config import *
from config_file.supernet_config import *

from lib import *
from search_strategy import *

if __name__ == "__main__":
    args = get_init_config("evaluate")

    logger = get_logger(args.logger_path)
    writer = get_writer(args.title, args.random_seed, args.writer_path)

    if args.seed is not None:
        logging.info("Set random seed : {}".format(args.seed))
        set_random_seed(args.seed)
    
    if args.cuda:
        device = torch.device("cuda" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    else:
        device = torch.device("cpu")

    architecture = load_architecture()

    macro_cfg, micro_cfg = get_supernet_cfg(args.search_space, args.classes)
    model = Model(args.macro_cfg, args.micro_cfg, architecture, args.classes, args.dataset)
    
    train_loader, val_loader = get_train_loader(args.dataset, args.dataset_path, args.batch_size, args.num_workers, train_portion=args.train_portion)

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

    if device.type == "cuda" and args.ngpu >= 1):
        model = model.to(device)
        model = nn.DataParallel(model, list(range(args.ngpu)))

    trainer = Trainer(criterion, optimizer, args.epochs, writer, logger, args.device, trainer_state="evaluate")

    start_time = time.time()
    trainer.train_loop(model, train_loader, val_loader)
    logger.info("Total search time : {:.2f}".format(time.time() - start_time))




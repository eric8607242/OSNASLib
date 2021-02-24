# Inference the architecture searched by NAS algorithm
import time

import torch

from config_file.arg_config import *
from config_file.supernet_config import *

from lib import *
from search_strategy import *

if __name__ == "__main__":
    args = get_init_config("evaluate")

    logger = get_logger(args.logger_path)
    writer = get_writer(args.title, args.random_seed, args.writer_path)

    if args.seed is not None:
        set_random_seed(args.seed)

    device = torch.device(args.device)

    architecture = load_architecture()

    macro_cfg, micro_cfg = get_supernet_cfg(
    args.search_space, args.classes, args.dataset)
    model = Model(
    args.macro_cfg,
    args.micro_cfg,
    architecture,
    args.classes,
     args.dataset)

    test_loader = get_test_loader(
    args.dataset,
    args.dataset_path,
    args.batch_size,
     args.num_workers)
    criterion = get_criterion()

    if device.type == "cuda" and args.ngpu >= 1):
        model=model.to(device)
        model=nn.DataParallel(model, list(range(args.ngpu)))

    trainer=Trainer(
    criterion,
    None,
    None,
    writer,
    logger,
    args.device,
    "evaluate",
     args)

    start_time=time.time()
    top1_acc=trainer.validate(model, test_loader, 0)
    logger.info("Final Top1 accuracy : {}".format(top1_acc))
    logger.info("Total search time : {:.2f}".format(time.time() - start_time))

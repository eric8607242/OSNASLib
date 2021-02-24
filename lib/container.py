import os
import time

import torch
import torch.nn as nn

from .config_file.arg_config import *
from .config_file.supernet_config import *

from . import *


class MetaContainer:
    def __init__(self, args):
        self.args = args

        logger = get_logger(args.logger_path)
        writer = get_writer(args.title, args.random_seed, args.writer_path)

        if args.random_seed is not None:
            set_random_seed(args.random_seed)

        device = torch.device(args.device)


class Searcher(MetaContainer):
    def __init__(self, args):
        super(Searcher, self).__init__(args)

        macro_cfg, micro_cfg = get_supernet_cfg(
            args.search_space, args.classes, args.dataset)
        supernet = Supernet(macro_cfg, micro_cfg, args.classes,
                            args.dataset, args.search_strategy,
                            bn_momentum=args.bn_momentum,
                            bn_track_running_stats=args.bn_track_running_stats)

        self.supernet = supernet.to(device)

        self.train_loader, self.val_loader = get_train_loader(
            self.args.dataset, self.args.dataset_path, self.args.batch_size, self.args.num_workers, train_portion=self.args.train_portion)

        self.training_strategy = TrainingStrategy(
            self.args.sample_strategy, len(micro_cfg), len(
                macro_cfg["search"]), self.supernet)

        self.lookup_table = LookUpTable(
            macro_cfg,
            micro_cfg,
            self.args.lookup_table_path,
            self.args.input_size,
            info_metric=[
                "flops",
                "param"])

        self.optimizer = get_optimizer(
            self.supernet.parameters(),
            self.args.optimizer,
            learning_rate=self.args.lr,
            weight_decay=self.args.weight_decay,
            logger=self.logger,
            momentum=self.args.momentum,
            alpha=self.args.alpha,
            beta=self.args.beta)

        self.lr_scheduler = get_lr_scheduler(
            self.optimizer,
            self.args.lr_scheduler,
            self.logger,
            step_per_epoch=len(
                self.train_loader),
            step_size=self.args.decay_step,
            decay_ratio=self.args.decay_ratio,
            total_epochs=self.args.epochs)

        self.criterion = get_criterion()

        start_epoch = 0
        if self.args.resume:
            start_epoch = resume_checkpoint(
                self.supernet,
                self.args.resume,
                self.optimizer,
                self.lr_scheduler)
            logger.info(
                "Resume training from {} at epoch {}".format(
                    self.args.resume, start_epoch))

        if self.device.type == "cuda" and self.args.ngpu >= 1:
            self.supernet = nn.DataParallel(
                self.supernet, list(range(self.args.ngpu)))

        self.trainer = Trainer(
            self.criterion,
            self.optimizer,
            self.lr_scheduler,
            self.writer,
            self.logger,
            self.args.device,
            "search",
            self.args,
            training_strategy=self.training_strategy,
            search_strategy=self.search_strategy,
            start_epoch=start_epoch)

    def search(self):
        start_time = time.time()
        self.logger.info("Searching process start!")

        if not self.args.directly_search:
            self.trainer.train_loop(
                self.supernet,
                self.train_loader,
                self.val_loader)

        best_architecture, best_architecture_hc, best_architecture_top1 = self.search_strategy.search(
            self.trainer, self.training_strategy)

        logger.info("Best architectrue : {}".format(best_architecture))
        logger.info(
            "Best architectrue top1 : {:.3f}".format(
                best_architecture_top1 * 100))
        logger.info("Best architectrue hc : {}".format(best_architecture_hc))

        save_architecture(self.args.searched_model_path, best_architecture)
        logger.info(
            "Total search time : {:.2f}".format(
                time.time() - start_time))


class Evaluator(MetaContainer):
    def __init__(self, args):
        super(Evaluator, self).__init__(args)

        architecture = load_architecture()

        macro_cfg, micro_cfg = get_supernet_cfg(
            self.args.search_space, self.args.classes, self.args.dataset)

        self.model = Model(
            self.args.macro_cfg,
            self.args.micro_cfg,
            architecture,
            self.args.classes,
            self.args.dataset)

        self.test_loader = get_test_loader(
            self.args.dataset,
            self.args.dataset_path,
            self.args.batch_size,
            self.args.num_workers)
        self.criterion = get_criterion()

        if device.type == "cuda" and self.args.ngpu >= 1:
            self.model = self.model.to(self.device)
            self.model = nn.DataParallel(
                self.model, list(range(self.args.ngpu)))

        self.trainer = Trainer(
            self.criterion,
            None,
            None,
            self.writer,
            self.logger,
            self.args.device,
            "evaluate",
            self.args)

    def evaluate(self):
        start_time = time.time()
        top1_acc = self.trainer.validate(self.model, self.test_loader, 0)
        logger.info("Final Top1 accuracy : {}".format(top1_acc))
        logger.info(
            "Total search time : {:.2f}".format(
                time.time() - start_time))

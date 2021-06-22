import sys
import os
import logging
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter


class AverageMeter(object):
    def __init__(self, name=''):
        self._name = name
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def __str__(self):
        return "%s: %.5f" % (self._name, self.avg)

    def get_avg(self):
        return self.avg

    def __repr__(self):
        return self.__str__()


def set_random_seed(seed):
    import random
    logging.info("Set seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(log_dir=None):
    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    log_format = "%(asctime)s | %(message)s"

    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt="%m/%d %I:%M:%S %p")

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "logger"))
        file_handler.setFormatter(logging.Formatter(log_format))

    logging.getLogger().addHandler(file_handler)
    return logger


def get_writer(title, seed, writer_dir=None):
    today = datetime.today()
    current_time = today.strftime("%d%m%Y%H%M%S")
    writer_dir = os.path.join(
        writer_dir,
        current_time +
        "_{}_{}".format(
            title,
            seed))

    writer = SummaryWriter(log_dir=writer_dir)
    return writer


def accuracy(output, target, topk=(1,)):
    """Compute the precision for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def resume_checkpoint(
        model,
        checkpoint_path,
        criterion=None,
        optimizer=None,
        lr_scheduler=None):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            if criterion is not None and "criterion" in checkpoint:
                criterion.load_state_dict(checkpoint["criterion"])

            if optimizer is not None and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])

            if lr_scheduler is not None and "scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["scheduler"])

            if "epoch" in checkpoint:
                resume_epoch = checkpoint["epoch"]

        else:
            model.load_state_dict(checkpoint)
    else:
        raise

    return resume_epoch


def save(
        model,
        checkpoint_path,
        criterion=None,
        optimizer=None,
        lr_scheduler=None,
        resume_epoch=None):
    if optimizer is None and resume_epoch is None and lr_scheduler is None:
        checkpoint = model.module.state_dict() if isinstance(
            model, nn.DataParallel) else model.state_dict()
    else:
        checkpoint = {"model": model.module.state_dict() if isinstance(
            model, nn.DataParallel) else model.state_dict()}
        if criterion is not None:
            checkpoint["criterion"] = criterion.state_dict()

        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()

        if lr_scheduler is not None:
            checkpoint["scheduler"] = lr_scheduler.state_dict()

        if resume_epoch is not None:
            checkpoint["epoch"] = resume_epoch

    torch.save(checkpoint, checkpoint_path)

def min_max_normalize(min_value, max_value, value):
    new_value = (value - min_value) / (max_value - min_value)
    return new_value

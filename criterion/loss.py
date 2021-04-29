import torch
import torch.nn.functional as F

def get_hc_criterion():
    return l2_hc_loss


def cross_entropy_with_label_smoothing(pred, target, eta=0.1):
    onehot_target = label_smoothing(pred, target, eta=eta)
    return cross_entropy_for_onehot(pred, onehot_target)


def label_smoothing(pred, target, eta=0.1):
    """
    Reference : https://arxiv.org/pdf/1512.00567.pdf
    """
    n_class = pred.size(1)
    target = torch.unsqueeze(target, 1)

    onehot_target = torch.zeros_like(pred)
    onehot_target.scatter_(1, target, 1)
    return onehot_target * (1 - eta) + eta / n_class * 1


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def l2_hc_loss(search_hc, target_hc, hc_weight):
    return (search_hc - target_hc) ** 2 * hc_weight

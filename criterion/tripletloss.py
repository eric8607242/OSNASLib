from itertools import combinations
from abc import ABC, abstractmethod

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineTripletLoss(nn.Module):
    """Triplet loss with associated triplet selector"""
    def __init__(self, criterion_config):
        super().__init__()
        self.criterion_config = criterion_config

        self.margin = self.criterion_config["margin"]
        self.triplet_selector = RandomNegativeTripletSelector(margin=self.margin)

    def forward(self, embeddings, labels):
        """Compute triplet loss

        Arguments:
            - embeddings (torch.Tensor): embeddings of shape (batch, embedding_dim)
            - labels (torch.LongTensor): target labels shape (batch,)

        Return:
            average triplet loss
        """
        target_embeddings = embeddings.detach().cpu().numpy()
        target_labels = labels.detach().cpu().numpy()
        triplets = self.triplet_selector.get_triplets(target_embeddings, target_labels)
        triplets = torch.LongTensor(triplets)
        triplets = triplets.to(embeddings.device)

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean()

def pdist(embeddings):
    term1 = np.power(embeddings, 2).sum(1).reshape(1, -1)
    term2 = -2 * np.dot(embeddings, embeddings.T)
    term3 = np.power(embeddings, 2).sum(1).reshape(-1, 1)

    return term1 + term2 + term3

class TripletSelector(ABC):
    """Select triplet samples from a sequence of embeddings and labels"""
    def __init__(self):
        pass

    @abstractmethod
    def get_triplets(self, embeddings, labels):
        """Select valid triplet samples

        Arguments:
            embeddings (np.ndarray): data of shape (batch, embedding_dim)
            labels (np.ndarray): data of shape (batch,)

        Return:
            numpy array of shape (valid_samples, 3)
            The dimension of 1 indicates the indices of (anchor, pos, neg) samples
        """
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):

    def __init__(self, margin, negative_selection_fn):
        super().__init__()
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        distance_matrix = pdist(embeddings) # batch, batch

        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue

            # Mask out negative samples
            negative_indices = np.where(np.logical_not(label_mask))[0]

            # Extract anchor_positive distances
            anchor_positives = list(combinations(label_indices, 2))
            anchor_positives = np.array(anchor_positives)
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]

            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                # Extract anchor_negatives distances
                an_distances = distance_matrix[np.array([anchor_positive[0]]), negative_indices]
                loss_values = ap_distance - an_distances + self.margin

                # Get triplet samples
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positives[0], anchor_positives[1], negative_indices[0]])

        triplets = np.array(triplets)
        return triplets


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

def HardestNegativeTripletSelector(margin):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=hardest_negative)

def RandomNegativeTripletSelector(margin):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=random_hard_negative)

def SemihardNegativeTripletSelector(margin):
    return FunctionNegativeTripletSelector(margin=margin, negative_selection_fn=lambda x: semihard_negative(x, margin))

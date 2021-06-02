import numpy as np
from sklearn.model_selection import KFold

def evaluate_roc(thresholds, embeddings_1, embeddings_2, labels, n_folds=10):
    fold_pairs = min(len(labels), embeddings_1.shape[0])
    fold_thresholds = len(thresholds)

    k_fold = KFold(n_splits=n_folds, shuffle=False)

    true_positive_rates = np.zeros((n_folds, fold_thresholds))
    false_positive_rates = np.zeros((n_folds, fold_thresholds))
    accuracy = np.zeros((n_folds))
    best_thresholds = np.zeros((n_folds))
    indices = np.arange(fold_pairs)

    distance = np.sum(np.square(np.subtract(embeddings_1, embeddings_2)), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        acc_train = np.zeros((fold_thresholds))
        for thr_idx, thr in enumerate(thresholds):
            _, _, acc_train[thr_idx] = evaluate_accuracy(
                thr, distance[train_set], labels[train_set])

        best_thr_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_thr_index]

        for thr_idx, thr in enumerate(thresholds):
            true_positive_rates[fold_idx, thr_idx], false_positive_rates[fold_idx,
                                                                         thr_idx], _ = evaluate_accuracy(thr, distance[test_set], labels[test_set])

        _, _, accuracy[fold_idx] = evaluate_accuracy(
            thresholds[best_thr_index], distance[test_set], labels[test_set])

    true_positive_rate = np.mean(true_positive_rates, 0)
    false_positive_rate = np.mean(false_positive_rates, 0)

    return true_positive_rate, false_positive_rate, accuracy, best_thresholds


def evaluate_accuracy(threshold, distance, labels):
    true_positive = np.sum(distance[labels == 1] < threshold, dtype=np.float)
    false_positive = np.sum(distance[labels == 0] < threshold, dtype=np.float)

    true_negative = np.sum(distance[labels == 0] > threshold, dtype=np.float)
    false_negative = np.sum(distance[labels == 1] > threshold, dtype=np.float)

    true_positive_rate = 0 if (true_positive + false_negative ==
                               0) else true_positive / (true_positive + false_negative)
    false_positive_rate = 0 if (false_positive + true_negative ==
                                0) else false_positive / (false_positive + true_negative)

    accuracy = (true_positive + true_negative) / distance.shape[0]

    return true_positive_rate, false_positive_rate, accuracy


def evaluate(embeds1, embeds2, labels, n_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    TP_ratio, FP_ratio, acc, best_thresholds = evaluate_roc(thresholds, embeds1, embeds2, labels, n_folds)
    return TP_ratio, FP_ratio, acc, best_thresholds

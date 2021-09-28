"""
Utility file for obtaining metrics for classifiers.

Peter Lais, 09/27/2021
"""

import torch
from sklearn.metrics import matthews_corrcoef, roc_auc_score

def summary_statistics(y_true, y_pred):
    """
    Returns the accuracy, MCC, and AUC for a given set of ground-truth
    and prediction data.

    Parameters
    ----------
    y_true: a 1D array containing the ground-truth class labels.
    y_pred: a 2D array whose columns correspond to the total number of
            classes. Each row's values should sum to one.

    Returns
    -------
    ACC, MCC, and AUC (AUROC) organized into a tuple.
    """

    assert y_true.ndim == 1 and y_pred.ndim == 2

    y_greedy = y_pred.argmax(-1)
    acc = torch.sum(y_greedy == y_true) / len(y_true)
    mcc = matthews_corrcoef(y_true, y_greedy)
    auc = roc_auc_score(y_true, y_pred if y_greedy.max() > 1 else y_pred.max(1)[0])

    return acc, mcc, auc

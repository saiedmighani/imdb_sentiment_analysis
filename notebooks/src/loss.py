import torch
import torch.nn as nn
import numpy as np


class LengthWeightedBCELoss(nn.Module):
    """
    Custom loss function for sentiment analysis that weights samples differently based on corpus document size.
    """
    def __init__(self):
        super().__init__()
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets, lengths):
        base_loss = self.base_loss(predictions, targets)
        weights = lengths / lengths.max()
        weighted_loss = base_loss * weights
        return weighted_loss.mean()

class VanillaBCELoss(nn.Module):
    """
    Vanilla BCE loss function for sentiment analysis.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets, lengths=None):
        return self.loss_fn(predictions, targets)


class ClassifierMetrics:
    """
    This class returns the accuracy, precision, and F1 metrics for a binary classifier
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def compute(self, labels, preds):
        preds = np.array([1 if p >= self.threshold else 0 for p in preds])
        labels = np.array(labels)

        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
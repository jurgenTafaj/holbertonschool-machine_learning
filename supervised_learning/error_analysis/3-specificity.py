#!/usr/bin/env python3
"""A function that calculates the specificity for each class
in a confusion matrix"""
import numpy as np


def specificity(confusion):
    """A function that calculates the specificity for each class
    in a confusion matrix"""
    classes = confusion.shape[0]
    spec = np.zeros(classes)
    for i in range(classes):
        true_negatives = np.sum(np.delete(np.delete(confusion, i, axis=0),
                                          i, axis=1))
        false_positives = np.sum(np.delete(confusion, i, axis=0)[:, i])
        spec[i] = true_negatives / (true_negatives + false_positives)
    return spec

#!/usr/bin/env python3
"""A function that calculates the sensitivity for each class
in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """A function that calculates the sensitivity for each
    class in a confusion matrix"""
    num_classes = confusion.shape[0]
    sensitivity_arr = np.zeros(num_classes)
    for i in range(num_classes):
        true_positives = confusion[i, i]
        false_negatives = np.sum(confusion[i, :]) - true_positives
        sensitivity_arr[i] = true_positives / (true_positives
                                               + false_negatives)
    return sensitivity_arr

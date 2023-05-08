#!/usr/bin/env python3
"""Convert one-hot labels and logits to class indices"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Convert one-hot labels and logits to class indices"""
    m, classes = labels.shape
    confusion = np.zeros((classes, classes))
    predicted_labels = np.argmax(logits, axis=1)
    correct_labels = np.argmax(labels, axis=1)
    for i in range(m):
        confusion[correct_labels[i], predicted_labels[i]] += 1
    return confusion

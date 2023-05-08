#!/usr/bin/env python3
"""A function that calculates the F1 score of a confusion matrix"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """A function that calculates the F1 score of a confusion
    matrix"""
    precisions = precision(confusion)
    sensitivities = sensitivity(confusion)
    return 2 * (precisions * sensitivities) / (precisions
                                               + sensitivities)

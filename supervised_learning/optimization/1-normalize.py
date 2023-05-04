#!/usr/bin/env python3
"""A function that normalizes a matrix"""


def normalize(X, m, s):
    """
    A function that normalizes a matrix
    """
    X_norm = (X - m) / s

    return X_norm

#!/usr/bin/env python3
"""Converts a one-hot matrix into a vector of labels"""
import numpy as np


def one_hot_encode(Y, classes):
    """Converts a one-hot matrix into a vector of labels"""
    try:
        assert isinstance(Y, np.ndarray)
        assert isinstance(classes, int)
        assert len(Y.shape) == 1
        assert np.all((Y >= 0) & (Y < classes))
        m = len(Y)
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None

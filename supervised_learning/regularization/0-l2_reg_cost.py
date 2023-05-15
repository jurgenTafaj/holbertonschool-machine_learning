#!/usr/bin/env python3
"""
A function that calculates the cost of a neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    A function that calculates the cost of a neural network
    with L2 regularization
    """
    l2_norm_squared = 0
    for l in range(1, L+1):
        l2_norm_squared += np.sum(np.square(weights["W" + str(l)]))
    l2_regularization_cost = (lambtha / (2 * m)) * l2_norm_squared
    return cost + l2_regularization_cost

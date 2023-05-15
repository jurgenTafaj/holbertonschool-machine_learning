#!/usr/bin/env python3
"""
A function that conducts forward propagation using Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    A function that conducts forward propagation using Dropout
    """
    cache = {}
    cache["A0"] = X
    for i in range(1, L + 1):
        Z = np.dot(weights["W{}".format(i)],
                   cache["A{}".format(
                    i - 1)]) + weights["b{}".format(i)]
        if i != L:
            tanh = np.sinh(Z) / np.cosh(Z)
            D1 = np.random.rand(tanh.shape[0], tanh.shape[1])
            D1 = (D1 < keep_prob).astype(int)
            tanh = tanh * D1
            tanh = tanh / keep_prob
            cache["D{}".format(i)] = D1
        else:
            t = np.exp(Z)
            tanh = np.exp(Z) / np.sum(t, axis=0, keepdims=True)
        cache["A{}".format(i)] = tanh
    return cache

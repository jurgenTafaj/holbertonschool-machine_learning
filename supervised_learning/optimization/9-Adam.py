#!/usr/bin/env python3
"""
A function that updates a variable in place
using the Adam optimization algorithm
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    A function that updates a variable in place
    using the Adam optimization algorithm
    """
    vdW = (beta1 * v) + ((1 - beta1) * grad)
    sdW = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    vdWc = vdW / (1 - (beta1 ** t))
    sdWc = sdW / (1 - (beta2 ** t))
    var -= alpha * (vdWc / (epsilon + (sdWc ** (1 / 2))))
    return var, vdW, sdW

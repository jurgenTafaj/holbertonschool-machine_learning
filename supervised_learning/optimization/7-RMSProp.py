#!/usr/bin/env python3
"""
A function that updates a variable using the RMSProp optimization algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    A function that updates a variable using the RMSProp optimization algorithm
    """
    new_moment = beta2 * s + (1 - beta2) * grad ** 2
    updated = var - alpha * grad / (new_moment ** (1 / 2) + epsilon)
    return updated, new_moment

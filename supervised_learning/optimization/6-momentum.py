#!/usr/bin/env python3
"""
A function that creates the training operation for a neural network in
tensorflow using the gradient descent with momentum optimization algorithm
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    A function that creates the training operation for a neural network in
    tensorflow using the gradient descent with momentum optimization algorithm
    """
    return tf.train.MomentumOptimizer(alpha, momentum=beta1).minimize(loss)

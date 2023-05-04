#!/usr/bin/env python3
"""
A function that creates the training operation for a neural network
in tensorflow using the RMSProp optimization algorithm
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    A function that creates the training operation for a neural network
    in tensorflow using the RMSProp optimization algorithm
    """
    rms = tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon)
    return rms.minimize(loss)

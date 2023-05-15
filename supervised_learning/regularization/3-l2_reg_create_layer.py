#!/usr/bin/env python3
"""
A function that creates a tensorflow layer that includes L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    A function that creates a tensorflow layer that includes 12
    regularization"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=regularizer)
    output = layer(prev)
    return output

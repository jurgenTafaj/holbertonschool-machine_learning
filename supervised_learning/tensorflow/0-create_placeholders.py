#!/usr/bin/env python3
"""
Module to create a placeholder
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    a function that create placeholders
    """
    return tf.placeholder(float, shape=[None, nx], name='x'), tf.placeholder(
        float, shape=[None, classes], name='y')

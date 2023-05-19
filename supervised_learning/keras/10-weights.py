#!/usr/bin/env python3
"""
Saves and load Model for Keras
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saves an entire model
    """
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    loads an entire model
    """
    network.load_weights(filename)
    return None

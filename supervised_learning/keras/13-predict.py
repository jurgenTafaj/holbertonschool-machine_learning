#!/usr/bin/env python3
"""
Tests a Neural Network
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    a function that tests a NN
    """
    return network.predict(data, verbose=verbose)

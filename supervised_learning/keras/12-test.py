#!/usr/bin/env python3
"""
Tests a Neural Network
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    a function that tests a NN
    """
    return network.evaluate(data, labels, verbose=verbose)

#!/usr/bin/env python3
"""
Saves and load Model for Keras
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves an entire model
    """
    # serialize model to json
    json_network = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_network)
    return None


def load_config(filename):
    """
    loads an entire model
    """
    with open(filename, 'r') as f:
        json_saved = f.read()
    return K.models.model_from_json(json_saved)

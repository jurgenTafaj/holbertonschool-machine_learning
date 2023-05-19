#!/usr/bin/env python3
"""Updates train model with early stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    a function that trains the model using early stopping
    """
    stopping = []
    if early_stopping and validation_data:
        stopping.append(K.callbacks.EarlyStopping(patience=patience))
    history = network.fit(data, labels, epochs=epochs, batch_size=batch_size,
                          shuffle=shuffle, verbose=verbose, callbacks=stopping,
                          validation_data=validation_data)
    return history

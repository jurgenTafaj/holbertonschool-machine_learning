#!/usr/bin/env python3
"""Deep neural architecture"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Dense block"""
    prev = X
    for i in range(layers):
        L = K.layers.BatchNormalization(axis=3)(prev)
        L = K.layers.Activation('relu')(L)
        L = K.layers.Conv2D(filters=growth_rate * 4,
                            kernel_size=(1, 1),
                            padding='same',
                            strides=(1, 1),
                            kernel_initializer='he_normal')(L)
        L = K.layers.BatchNormalization(axis=3)(L)
        L = K.layers.Activation('relu')(L)
        L = K.layers.Conv2D(filters=growth_rate,
                            kernel_size=(3, 3),
                            padding='same',
                            strides=(1, 1),
                            kernel_initializer='he_normal')(L)
        prev = K.layers.concatenate([prev, L], axis=3)
        nb_filters = nb_filters + growth_rate
    return(prev, nb_filters)

#!/usr/bin/env python3
"""Performs convolution on grayscale images with padding """


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs convolution on grayscale images with padding"""

    m, height, width = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    ch = height + 2 * ph - kh + 1
    cw = width + 2 * pw - kw + 1

    p_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))

    convoluted = np.zeros((m, ch, cw))

    for h in range(ch):
        for w in range(cw):
            output = np.sum(p_images[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted

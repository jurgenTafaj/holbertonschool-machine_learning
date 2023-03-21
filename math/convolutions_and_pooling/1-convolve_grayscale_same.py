#!/usr/bin/env python3
"""Performs a same convolution on grayscale images"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""

    m, height, width = images.shape
    kh, kw = kernel.shape
    if kh % 2 != 0 and kw % 2 != 0:
        padding_height = (kh - 1) // 2
        padding_width = (kw - 1) // 2
    else:
        padding_height = kh // 2
        padding_width = kw // 2

    p_images = np.pad(images, ((0, 0), (padding_height, padding_height),
                      (padding_width, padding_width)))

    convoluted = np.zeros((m, height, width))

    for h in range(height):
        for w in range(width):
            output = p_images[:, h: h + kh, w: w + kw] * kernel
            sum_out = np.sum(output, axis=(1, 2))
            convoluted[:, h, w] = sum_out
    return convoluted

#!/usr/bin/env python3
"""Performs a convolution on images with nc kernels"""


import numpy as np


def convolve(images, kernels, padding="same", stride=(1, 1)):
    """Performs a convolution on images with nc kernels"""
    m, height, width, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if padding == "same":
        ph = ((height - 1) * sh + kh - height) // 2 + 1
        pw = ((width - 1) * sw + kw - width) // 2 + 1
    elif padding == "valid":
        ph = 0
        pw = 0
    else:
        ph, pw = padding

    p_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    ch = ((height + 2 * ph - kh) // sh) + 1
    cw = ((width + 2 * pw - kw) // sw) + 1

    convoluted = np.zeros((m, ch, cw, nc))

    for h in range(ch):
        for w in range(cw):
            for k in range(nc):
                output = (
                    p_images[:, h * sh: h * sh + kh, w * sw: w * sw + kw]
                    * kernels[:, :, :, k]
                )
                sum_out = np.sum(output, axis=(1, 2, 3))
                convoluted[:, h, w, k] = sum_out
    return convoluted

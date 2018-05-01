import numpy as np
import numpy.matlib
import math

def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    padded_im = np.zeros((Hi + Hk - 1, Wi + Wk - 1))
    row_pad = int((Hk - 1) / 2)
    col_pad = int((Wk - 1) / 2)
    padded_im[row_pad:-row_pad, col_pad:-col_pad] = np.copy(image)
    flip_kernel = np.flip(np.flip(kernel, 0), 1)

    for m in range(Hi):
        for n in range(Wi):
            conv_sum = 0
            for i in range(Hk):
                for j in range(Wk):
                    conv_sum  += flip_kernel[i, j] * padded_im[m+i, n+j]
            out[m, n] = conv_sum

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    out = np.zeros((H + 2*pad_height, W + 2*pad_width))
    out[pad_height:-pad_height, pad_width:-pad_width] = np.copy(image)

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    padded_im = zero_pad(image, math.ceil((Hk-1)/2), math.ceil((Wk-1)/2))
    f_kernel  = np.flip(np.flip(kernel, 0), 1)

    for m in range(Hi):
        for n in range(Wi):
            out[m, n] = np.sum(np.multiply(f_kernel, padded_im[m:m+Hk, n:n+Wk]))

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    Hgap = int((Hk - 1)/2)
    Wgap = int((Wk - 1)/2)
    I = np.zeros((Hi+Hk-1, Wi+Wk-1))
    I[:Hi, :Wi] = np.copy(image)
    K = np.zeros((Hi+Hk-1, Wi+Wk-1))
    K[:Hk, :Wk] = np.copy(kernel)

    I = np.fft.fft2(I)
    K = np.fft.fft2(K)

    C   = np.multiply(I, K)
    out = np.real(np.fft.ifft2(C))[Hgap:Hi+Hgap, Wgap:Wi+Wgap]

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    out = conv_fast(f, np.flip(np.flip(g, 0), 1))

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g
    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    out = cross_correlation(f, g) - np.mean(g)

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    Hg, Wg = g.shape
    Hf, Wf = f.shape
    out = np.zeros((Hf, Wf))
    padded_f = zero_pad(f, math.ceil((Hg-1)/2), math.ceil((Wg-1)/2))
    g2 = (g-np.mean(g)) / np.std(g)

    for m in range(Hf):
        for n in range(Wf):
            fmn = padded_f[m:m+Hg, n:n+Wg]
            out[m, n] = np.sum(np.multiply(g2, (fmn - np.mean(fmn))/np.std(fmn)))

    return out

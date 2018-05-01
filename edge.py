import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # We will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    f_kernel  = np.flip(np.flip(kernel, 0), 1)

    for m in range(Hi):
        for n in range(Wi):
            out[m, n] = np.sum(np.multiply(f_kernel, padded[m:m+Hk, n:n+Wk]))

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.
    
    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """  
    
    kernel = np.zeros((size, size))

    edge_val = (size - 1) // 2
    I, J = np.meshgrid(np.arange(-edge_val, edge_val+1), np.arange(-edge_val, edge_val+1), indexing = 'ij')

    kernel = (1/(2 * np.pi * sigma**2)) * np.exp( - (np.square(I) + np.square(J)) / (2*sigma**2) )

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    kernel = np.array([[0.5, 0.0, -0.5]])
    out = conv(img, kernel)

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    kernel = np.array([[0.5], [0.0], [-0.5]])
    out = conv(img, kernel)

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    DGx = partial_x(img)
    DGy = partial_y(img)

    G = np.sqrt(np.square(DGx) + np.square(DGy))
    theta = ((np.arctan2(DGy, DGx) * 180 / np.pi) + 180) % 360

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    
    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))
    out = np.copy(G)

    # Round the gradient direction to the nearest 45 degrees
    theta = (np.floor(((theta + 22.5) % 360) / 45) * 45) % 180

    padded_im = np.zeros((H+2, W+2))
    padded_im[1:-1, 1:-1] = np.copy(G)

    #Iterate over all directions and subtract both sides from the center.
    #If the gradient direction matches update the value accordingly.
    grad_index = {0: (1, 2, 1, 0), 45: (0, 0, 2, 2), 90: (0, 1, 2, 1), 135:(0, 2, 2, 0)}
    for gdir in grad_index.keys():
        active1 = G - padded_im[grad_index[gdir][0]:H+grad_index[gdir][0], grad_index[gdir][1]:W+grad_index[gdir][1]]
        active2 = G - padded_im[grad_index[gdir][2]:H+grad_index[gdir][2], grad_index[gdir][3]:W+grad_index[gdir][3]]
        out[np.logical_and(theta == gdir, np.logical_or(active1 < 0, active2 < 0))] = 0

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    strong_edges = (img >= high)
    weak_edges   = np.logical_and(img < high, img > low)

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

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


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Check if a weak_edges pixel is adjacent to a strong one. If it is, that
    pixel becomes strong too. We keep iterating until the edge image stops
    changing (inefficient). A better approach would be to iterate over each
    pixel in strong_edges and perform breadth firstsearch across the connected
    pixels in weak_edges to link them. Here we consider a pixel (a, b) is 
    connected to a pixel (c, d)if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    upd_edges = np.zeros((H, W))
    upd_edges[strong_edges == 1] = 1
    kernel = np.ones((3, 3))

    while (np.any(np.logical_xor(edges, upd_edges))):
        edges = np.copy(upd_edges)
        strong_ind = conv_faster(edges, kernel)
        upd_edges[np.logical_and(weak_edges == 1, strong_ind > 0)] = 1

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    smoothed_img = conv(img, gaussian_kernel(kernel_size, sigma))
    G, theta     = gradient(smoothed_img)
    strong, weak = double_thresholding(non_maximum_suppression(G, theta), high, low)
    edge         = link_edges(strong, weak)

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)
        
    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    XY = np.stack((xs, ys)).T
    CS = np.stack((cos_t, sin_t))

    r = (np.floor((np.matmul(XY, CS)) + 0.5) + diag_len).astype(int)
    accumulator = np.apply_along_axis(lambda x: np.bincount(x, minlength = 2*diag_len+1), axis = 0, arr = r)

    return accumulator, rhos, thetas

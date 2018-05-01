# Computer-Vision-Projects
Projects in the field of Computer Vision and Image Processing

1 - Filters
The first tutorial project is on convolution. Here there are three implementations of 2-D convolution. The first is the naive one using four nested loops. The second uses 2 loops and numpy operations to apply faster element-wise computations. The third and fastest one is based on FFT.

2 - Edge recognition (Canny + Hough Transform)
Implmemtation of the Canny edge detector. First, the image gradient is computed and non-maximum suppression is applied on every pixel along the direction of the gradient at that point. With the use of two thresholds we categorize each pixel as a strong potential edge, a weak potential edge or as not an edge (discarded). We then update the weak edges. If a weak edge is connected to a strong edge then it's also promoted to a strong edge. The Canny edge detector the returns the strong edges as the real edges. We finally use those results in conjuction with the Hough transform to detect straight edges.

3 - Panorama + RANSAC
Implementation of a panorama application using a simple descriptor. We use the Harris corner detection algorithm to identify keypoints. We use a simple descriptor for each one. Using a threshold we find pairs of keypoints that are close to each other between the two images. Then an affine transformation matrix is calculated by solving a least-squares problem. We finally use teh RANSAC algorithm to get a better transormation. RANSAC picks random subsamples of the keypoont pairs and tries to find a transormation that best fits the closest points. Further improvememtns can be made by using better desciptors such as SIFT or SURF.

4 - Seam carving
Implementation of seam carving. We compute the energy of the image. This can be done in many different ways. Here we use the sum of the absolute values of the x and y gradients. We then compute the cost of each pixel, which is the sum of the energy aong the lowest enerrgy path from the top. The pixel at the final row with the lowest cost defines a path which is called the optimal seam (the lowest energy path from the top to the bottom pixel with the lowest cost). By iteratively removing the optimal seam from an image, we can reduce the size of the image without removing too much energy from it (usually located at edges and high-frequency content).

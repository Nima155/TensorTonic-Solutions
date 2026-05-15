import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    centroids = np.array(centroids)
    pts = np.zeros(len(points))
    for i, p in enumerate(points):
        mn = np.argmin(((p - centroids) ** 2).sum(axis = 1))
        pts[i] = mn
    return list(pts)
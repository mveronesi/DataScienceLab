import numpy as np
from tqdm import tqdm

def distance_matrix(X, metric=None):
    """Return a symmetric matrix representing distances between points
    :param X: input data points, array of shape (N, C)
    :param metric: function for computing distance between two points;
                   if None then euclidean distance is considered
    :return: distances: a symmetric matrix of shape (N, N)
    """
    N = X.shape[0] # number of samples
    C = X.shape[1] # number of dimensions
    if metric is None:
        # if there is no metric then the euclidean distance is considered
        metric = lambda x1, x2: np.linalg.norm((x1-x2), 2)
    distances = np.zeros((N, N), dtype=float)
    for i in tqdm(range(N)): # row scan
        for j in range(i): # column scan until the main diagonal (excluded) 
            distances[i, j] = distances[j, i] = metric(X[i, :], X[j, :])
    return distances
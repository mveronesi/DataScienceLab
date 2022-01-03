import numpy as np
from LinkedList import LinkedList
from tqdm import tqdm
from distance_matrix import distance_matrix

def compute_labels_hash(labels):
    labels_set = list(set(labels))
    labels_hash = [LinkedList(dtype=int) for _ in labels_set]
    for i in range(len(labels)):
        labels_hash[labels[i]].push_back(i)
    return labels_hash

def silhouette_samples(X, labels, distance_measure=None, labels_hash=None):
    """Evaluate the silhouette for each point and return them as a list.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : array shape = (N,)"""
    N = X.shape[0]
    C = X.shape[1]

    if labels_hash is None:
        labels_hash = compute_labels_hash(labels)
    if distance_measure is None:
        distance_measure = lambda x1, x2: np.linalg.norm(x1 - x2, 2)
    distances = distance_matrix(X, metric=distance_measure)

    silhouette = np.zeros(N)   
    for i in tqdm(range(N)):
        # COMPUTING a(i)
        cluster_points_i = labels_hash[labels[i]].to_array()
        a = 1 / (labels_hash[labels[i]].size - 1)
        acc = 0
        for j in cluster_points_i:
            acc += distances[i, j]
        a *= acc
        # COMPUTING b(i)
        b = None
        for c in range(len(labels_hash)):
            if c != labels[i]:
                acc = 0
                cluster_points_j = labels_hash[c].to_array()
                for j in cluster_points_j:
                    acc += distances[i, j]
                acc *= (1 / labels_hash[c].size)
                if b is None or acc < b:
                    b = acc

        # COMPUTING SILHOUETTE(i)
        silhouette[i] = (b - a) / max(a, b) if len(cluster_points_i) > 1 else 1
    return silhouette

def silhouette_score(X, labels, distance_measure=None, labels_hash=None):
    """Evaluate the silhouette for each point and return the mean.
    :param X: input data points, array, shape = (N,C).
    :param labels: the list of cluster labels, shape = N.
    :return: silhouette : float"""

    return silhouette_samples(X, labels, distance_measure, labels_hash).mean()
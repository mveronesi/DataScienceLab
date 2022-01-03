import random
import numpy as np
from LinkedList import LinkedList
from tqdm import tqdm
from matplotlib import pyplot as plt

class KMeans:
    def __init__(self, n_clusters, max_iter=100, distance_metric=None, stop_frac=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.labels_hash = None
        self.stop_frac = 1 / stop_frac if stop_frac is not None else 1 / 100
        self.distance_metric = lambda x1, x2: np.linalg.norm((x1-x2), 2) if distance_metric is None else distance_metric

    def fit_predict(self, X, plot_clusters=True, plot_step=5):
        """Run the K-means clustering on X
        :param X: input data points, array of shape (N, C)
        :return: labels: array, shape = (N,)
        """
        N = X.shape[0] # number of samples
        C = X.shape[1] # number of dimensions
        ############# KMEANS INITIALIZATION PROCEDURE ###################
        self.labels = np.zeros(N, dtype=int) - 1
        # generating initial centroids randomly
        initial_centroids_indexes = random.sample(range(N), self.n_clusters) # contains only indexes of centroids
        self.centroids = np.zeros((self.n_clusters, C,))
        for i in range(len(initial_centroids_indexes)):
            self.centroids[i, :] = X[initial_centroids_indexes[i], :]
        ########### END OF INITIALIZATION PROCEDURE ###################

        self.labels_hash = [LinkedList(dtype=int) for _ in range(self.n_clusters)]
        for k in tqdm(range(self.max_iter)):
            # assign all points to the closest centroid
            labels_changing = 0
            for i in range(N):
                nearest_centroid_dist = None
                nearest_centroid_label = None
                for j in range(self.centroids.shape[0]): # find nearest centroid
                    d = self.distance_metric(X[i, :], self.centroids[j, :])
                    if nearest_centroid_dist is None or d < nearest_centroid_dist:
                        nearest_centroid_dist = d
                        nearest_centroid_label = j
                if self.labels[i] != nearest_centroid_label:
                    self.labels[i] = nearest_centroid_label
                    labels_changing += 1
                self.labels_hash[nearest_centroid_label].push_back(i)

            if plot_clusters == True and k % plot_step == 0:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.scatter(X[:, 0], X[:, 1], c=self.labels)
                ax.scatter(self.centroids[:, 0], self.centroids[:, 1], color="red", marker="*")

            if labels_changing < (self.stop_frac):
                print("STOP: too few labels are changing")
                break
            # recompute centroids
            for i in range(self.n_clusters):
                # calculating mean of the current label
                point_indexes = self.labels_hash[i].to_array()
                n = 0
                tot = np.zeros(C)
                for j in point_indexes:
                    tot += X[int(j), :] # TODO check
                    n += 1
                self.centroids[i] = tot / n # centroid
        return self.labels

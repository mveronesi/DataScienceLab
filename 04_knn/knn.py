import numpy as np

def euclidean_distance(x1, x2):
        """Compute euclidean distance among two numpy arrays.
        Raise an error if param arrays have different shapes
        :param x1: numpy array of shape = (N, )
        :param x2: numpy array of shape = (N, )
        :return: float value, the euclidean distance between x1 and x2
        """
        if x1.ndim > 1 or x2.ndim > 1 or len(x1) != len(x2):
            raise ValueError
        return np.linalg.norm(x1-x2, ord=2)

def manhattan_distance(x1, x2):
    """Compute manhattan distance among two numpy arrays.
    Raise an error if param arrays have different shapes
    :param x1: numpy array of shape = (N, )
    :param x2: numpy array of shape = (N, )
    :return: float value, the euclidean distance between x1 and x2
    """
    if x1.ndim > 1 or x2.ndim > 1 or len(x1) != len(x2):
            raise ValueError
    return np.linalg.norm(x1-x2, ord=1)

def cosine_similarity(x1, x2):
    """Compute cosine similarity among two numpy arrays.
    Raise an error if param arrays have different shapes
    :param x1: numpy array of shape = (N, )
    :param x2: numpy array of shape = (N, )
    :return: float value, the euclidean distance between x1 and x2
    """
    if x1.ndim > 1 or x2.ndim > 1 or len(x1) != len(x2):
            raise ValueError
    num = np.sum(np.multiply(x1, x2))
    den = np.linalg.norm(x1, 2) * np.linalg.norm(x2, 2)
    return num / den

def accuracy(predict, real):
    """Compute value of accuracy for predicted values.

    :param predict: list of predicted values shape = (N, )
    :param real: list of real values shape = (N, 1)
    :return: float value representing percentage accuracy score
    """
    tot = len(real)
    corrects = 0
    for i in range(tot):
        if predict[i] == real[i]:
            corrects += 1
    return corrects / tot * 100
    
class kNN:
    def __init__(self, k, distance_metric='euclidean', weights='uniform'):
        if k >= 1:
            self.k = k
        else:
            self.k = None
            raise ValueError

        if distance_metric == 'euclidean':
            self.distance_metric = euclidean_distance
            self.nearest = self.nearest_distance
        elif distance_metric == 'cosine':
            self.distance_metric = cosine_similarity
            self.nearest = self.nearest_cosine
        elif distance_metric == 'manhattan':
            self.distance_metric = manhattan_distance
            self.nearest = self.nearest_distance
        else:
            self.distance_metric = None
            self.nearest = None
            raise ValueError

        if weights == 'uniform':
            self.majority_voting = self.majority_voting_uniform
        elif weights == 'distance':
            self.majority_voting = self.majority_voting_distance
        else:
            self.majority_voting = None
            raise ValueError


    def fit(self, x, y):
        """
        Store the 'prior knowledge' of you model that will be used
        to predict new labels.
        :param X : input data points, ndarray, shape = (R,C).
        :param y : input labels, ndarray, shape = (R,).
        """
        self.x = x
        self.y = y
        classes = set()
        for row in y:
            classes.add(row)
        self.classes = classes

    def majority_voting_uniform(self):
        """
        Compute the most frequent class in the training set slice composed
        by indexes passed
        :param indexes: list of indexes of candidates in training set that can vote
        """
        voting = dict()
        for i in self.knn_indexes:
            if self.y[i] in voting.keys():
                voting[self.y[i]] += 1
            else:
                voting[self.y[i]] = 1
        return max(zip(voting.values(), voting.keys()))[1]

    def majority_voting_distance(self):
        voting = dict()
        for i in self.knn_indexes:
            if self.y[i] in voting.keys():
                voting[self.y[i]] += 1/self.distances[i]
            else:
                voting[self.y[i]] = 1/self.distances[i]
        return max(zip(voting.values(), voting.keys()))[1]

    def nearest_distance(self, distances):
        return np.argpartition(distances, self.k)[:self.k]

    def nearest_cosine(self, distances):
        return np.argpartition(distances, -self.k)[-self.k:]

    def predict(self, x):
        """Run the KNN classification on X.
        :param X: input data points, ndarray, shape = (N,C).
        :return: labels : ndarray, shape = (N,).
        """
        y = []
        for i in x:
            distances = []
            for j in self.x:
                distances.append(self.distance_metric(i, j))
            self.distances = np.array(distances)
            self.knn_indexes = self.nearest(self.distances)
            y.append(self.majority_voting())
        return np.array(y)
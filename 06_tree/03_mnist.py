import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def majority_voting(a):
    """Return the most frequent class in a.

    - param a: np.array of shape (n, )
    - return: the most recurrent class in a.
    """
    a = np.array(a, dtype=int)
    counts = np.bincount(a)
    return np.argmax(counts)

def build_tree(X, y, N, max_features):
    index = np.random.choice(X.shape[0], N, replace=True)
    X_train = X[index]
    y_train = y[index]
    model = DecisionTreeClassifier(max_features=max_features)
    model.fit(X_train, y_train)
    return model

def build_random_forest(X, y, n_trees, max_features):
    print("Training with ", n_trees, " trees")
    model = MyRandomForestClassifier(n_trees, max_features=max_features)
    model.fit(X, y)
    return model

class MyRandomForestClassifier():
    def __init__(self, n_estimators, max_features=None, N=None, criteria="majority"):
        """Create a random forest model.

        - param n_estimators: number of trees in the random forest
        - param max_features: number of max features that a single tree can consider
        - param N: number of samples extracted with replacement from the dataset. If None N = number of samples in training set.
        - return: a model ready to be trained
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.N = N
        self.trees = list()
        if criteria == "majority":
            self.criteria = majority_voting
        else:
            raise ValueError("criteria parameter is not valid")

    def fit(self, X, y):
        """train the trees of this random forest using subsets of X (and y)"""
        N = self.N if self.N is not None else len(y)
        futures_list = []
        with ThreadPoolExecutor(max_workers=8) as exe:
            for _ in range(self.n_estimators):
                futures_list.append(exe.submit(build_tree, X, y, N, self.max_features))
        wait(futures_list, timeout=None, return_when=ALL_COMPLETED)
        self.trees = [i.result() for i in futures_list]
        
    def predict(self, X):
        """predict the label for each point in X"""
        y_preds = np.zeros((len(X), len(self.trees),))
        for i in range(len(self.trees)):
            y_preds[:, i] = np.array(self.trees[i].predict(X))
        y_pred = np.zeros(len(X))
        for i in range(len(y_pred)):
            y_pred[i] = self.criteria(y_preds[i, :])
        return y_pred


with open("../datasets/mnist", "rb") as f:
        dataset = pickle.load(f)

X = np.array(dataset["data"], dtype=int)
y = np.array(dataset["target"], dtype=int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/7))

model = MyRandomForestClassifier(10, 28)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)
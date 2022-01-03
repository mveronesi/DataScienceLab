import numpy as np
import csv
from knn import kNN
from knn import accuracy
import matplotlib.pyplot as plt

dataset = []
with open("../datasets/mnist.csv") as f:
    for row in csv.reader(f):
        dataset.append(row)

test_indexes = np.random.choice(np.arange(len(dataset)), size=int(len(dataset)*0.2), replace=False)
x_test = []
y_test = []
x_train = []
y_train = []

sample_length = len(dataset[0])
features_indexes = np.random.choice(np.arange(1, sample_length), size=100, replace=False)

for i in range(len(dataset)):
    tmp = []
    for j in features_indexes:
        tmp.append(dataset[i][j])
    if i in test_indexes:
        x_test.append(np.array(tmp, dtype=int))
        y_test.append(int(dataset[i][0]))
    else:
        x_train.append(np.array(tmp, dtype=int))
        y_train.append(int(dataset[i][0]))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

k_values = np.arange(1, 11)
acc_values = []

for k in k_values:
    model = kNN(k, 'euclidean', 'uniform')
    model.fit(x_train, y_train)
    print("A")
    y_predict = model.predict(x_test)
    print("B")
    acc = accuracy(y_predict, y_test)
    acc_values.append(acc)

acc_values = np.array(acc_values)

plt.figure()
plt.plot(k_values, acc_values)
plt.show()
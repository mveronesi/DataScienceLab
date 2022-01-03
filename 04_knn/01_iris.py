import numpy as np
import csv
from knn import kNN
from knn import accuracy
import matplotlib.pyplot as plt

dataset = []
# load dataset
with open("../datasets/iris.csv") as f:
    for row in csv.reader(f):
        dataset.append(row)

test_indexes = np.random.choice(
    np.arange(len(dataset)), size=int(len(dataset)*0.2), replace=False)
x_test = []
y_test = []
x_train = []
y_train = []

for i in range(len(dataset)):
    tmp = []
    for j in range(4):
        tmp.append(dataset[i][j])
    if i in test_indexes:
        x_test.append(np.array(tmp, dtype=float))
        y_test.append(dataset[i][4])
    else:
        x_train.append(np.array(tmp, dtype=float))
        y_train.append(dataset[i][4])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

k_values = np.arange(1, 11)
acc_values = []

for k in k_values:
    model = kNN(k, 'euclidean', 'uniform')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    acc = accuracy(y_predict, y_test)
    acc_values.append(acc)

acc_values = np.array(acc_values)

plt.figure()
plt.plot(k_values, acc_values)
plt.show()

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

path = 'C:\\Users\\Administrator\\Desktop\\data.txt'
data = pd.read_csv(path, header=None)
plt.scatter(data[:][0], data[:][1], marker='+')
data = np.array(data)
m = data.shape[0]
theta = np.array([0, 0])
data = np.hstack([np.ones([m, 1]), data])
y = data[:, 2]
data = data[:, :2]


def cost_function(data, theta, y):
    cost = np.sum((data.dot(theta) - y) ** 2)
    return cost / (2 * m)


def gradient(data, theta, y):
    grad = np.empty(len(theta))
    grad[0] = np.sum(data.dot(theta) - y)
    for i in range(1, len(theta)):
        grad[i] = (data.dot(theta) - y).dot(data[:, i])
    return grad


def gradient_descent(data, theta, y, eta):
    while True:
        last_theta = theta
        grad = gradient(data, theta, y)
        theta = theta - eta * grad
        print(theta)
        if abs(cost_function(data, last_theta, y) - cost_function(data, theta, y)) < 1e-15:
            break
    return theta


res = gradient_descent(data, theta, y, 0.0001)
X = np.arange(3, 25)
Y = res[0] + res[1] * X
plt.plot(X, Y, color='r')
plt.show()

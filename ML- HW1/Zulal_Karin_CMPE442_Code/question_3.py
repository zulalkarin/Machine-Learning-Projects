import numpy as np
import matplotlib.pyplot as plt

m = 100
X = np.random.rand(m, 1) * 2
y = np.sin(2 * np.pi * X) + np.random.randn(m, 1)
plt.plot(X, y, "r.")

def computeWeights(X, testx, tau):
    weights = np.exp(-((X-testx)*(X-testx))/(2*tau**2))
    return weights

def weighted_linear_regression(X, y, iterNo, testx, eta, tau):
    theta = np.random.randn(2, 1) # initialize theta with random values
    row, col = np.shape(X) # num of rows and columns in X
    weights = np.zeros((row, 1)) #empty array for store weights

    for i in range(row): # compute the weights for training examples
        weights[i] = computeWeights(X[:, 1][i], testx[1], tau)

    for i in range(iterNo):
        grad = (2 / m) * (weights * X).T.dot(X.dot(theta) - y) # compute the gradient using the weighted training examples
        theta = theta - eta * grad # update theta using the learning rate and gradient

    return theta



X_with_bias = np.c_[np.ones((m, 1)), X] #We use weighted linear regression so, I created this variable to adding a column

eta = 0.4
tau = 0.01
iteration_cnt = 100
y1_predict = np.zeros((m, 1))
i = 0
for x in X_with_bias:
    theta = weighted_linear_regression(X_with_bias, y, iteration_cnt, x, eta, tau)
    pred = x.dot(theta)
    y1_predict[i] = pred
    i = i + 1


sorted_zip = sorted(zip(X, y1_predict))
X, y1_predict = zip(*sorted_zip)
plt.plot(X, y1_predict, color='m')
plt.title('Tau ' + str(tau))
plt.show()

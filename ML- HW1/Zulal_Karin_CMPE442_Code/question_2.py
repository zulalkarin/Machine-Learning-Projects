import numpy as np
import matplotlib.pyplot as plt

m = 100
X = np.random.rand(m, 1)
y = 100 + 3 * X + np.random.randn(m, 1)
plt.plot(X, y, "r.")

def linear_regression(X, y, iterNo, eta):
    #m = np.shape[0]  # Number of training samples
    #X = np.hstack((np.ones((m, 1)), X))  # Adding intercept term to the data
    #n = np.shape[1]  # Number of features

    theta = np.random.randn(2, 1)
    MSE = np.zeros(iterNo)  # Array to store Mean Squared Error for every iteration

    for i in range(iterNo):
        y_predicted = X.dot(theta) - y # Compute the predicted values
        # Compute the gradient of the cost function with respect to theta
        gradient = 2 * X.T.dot(y_predicted) / m
        # Update theta using gradient descent
        theta -= eta * gradient
        MSE[i] = np.sum(y_predicted ** 2) * (1 / m)

    return theta, MSE



X_b = np.c_[np.ones((m, 1)), X]
theta, MSE = linear_regression(X_b, y, 1000, 0.5)
y_predict = X_b.dot(theta)

plt.plot(X, y_predict, 'b-')
plt.show()

print(f"h(x) = {theta[0][0]} + {theta[1][0]}x")

iterNo = np.arange(1000)
plt.plot(iterNo, MSE)
plt.show()

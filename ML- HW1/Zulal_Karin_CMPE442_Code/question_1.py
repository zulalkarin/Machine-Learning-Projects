import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Generate synthetic data
m = 100
X = np.random.rand(m, 1) * 2
y = np.sin(2*np.pi*X) + np.random.randn(m, 1)

# Fit polynomial regression models of different degrees and plot the results
degrees = [0, 1, 3, 9]
for i, d in enumerate(degrees):
    polynomial_features = PolynomialFeatures(degree=d)
    x_poly = polynomial_features.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    y_poly_pred = lin_reg.predict(x_poly)
    sorted_zip = sorted(zip(X, y_poly_pred))
    X, y_poly_pred = zip(*sorted_zip)
    plt.figure(i)
    plt.scatter(X, y, color='blue', label='data')
    plt.plot(X, y_poly_pred, color='red', label='degree ' + str(d))
    plt.legend()
    plt.title('Polynomial Regression (degree ' + str(d) + ')')
    plt.xlabel('x')
    plt.ylabel('y')

plt.show()

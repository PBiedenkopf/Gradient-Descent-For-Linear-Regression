""" 
    Gradient Descent Algorithm for linear regression of large datasets
    P. Biedenkopf - 11.12.2020

"""

from linear_regression import LinearRegression, PolynomialRegression
import numpy as np


# Load fish market dataset
dataset = './datasets/fish_market.csv'
data = np.genfromtxt(dataset, delimiter=',', skip_header=1)
y = data[:,1]
X = np.insert(data[:,2:], 0, data[:,0], axis=1)

# Compute linear regression model
print("Linear regression model")
model = LinearRegression()
model.train(X, y)
X_predict = np.array([1, 23.4, 26.2, 30.6, 11.2, 4.5])
predicted_y = model.predict(X_predict)

# Compute linear regression model with polynomial features
print("Regression model with polynomial features")
modelPoly = PolynomialRegression(Lambda=10, power=5)
modelPoly.train(X, y)
X_predict = np.array([1, 23.4, 26.2, 30.6, 11.2, 4.5])
predicted_y_poly = modelPoly.predict(X_predict)

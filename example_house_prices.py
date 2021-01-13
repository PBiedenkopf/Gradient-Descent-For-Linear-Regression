""" 
    Gradient Descent Algorithm for linear regression of large datasets
    P. Biedenkopf - 11.12.2020

"""

from linear_regression import LinearRegression, PolynomialRegression
import numpy as np



# Load house prices dataset
dataset = './datasets/house_prices.csv'
data = np.genfromtxt(dataset, delimiter=',', skip_header=1)
y = data[:, 2]
X = data[:,:2]

# Compute linear regression model
print("Linear regression model")
model = LinearRegression()
model.train(X, y)
X_predict = np.array([2000.0, 3.0])
predicted_y = model.predict(X_predict)

# Compute linear regression model with polynomial features
print("Regression model with polynomial features")
modelPoly = PolynomialRegression(Lambda=10, power=5)
modelPoly.train(X, y)
X_predict = np.array([2000.0, 3.0])
predicted_y_poly = modelPoly.predict(X_predict)

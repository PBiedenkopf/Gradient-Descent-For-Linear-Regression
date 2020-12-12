""" 
    Gradient Descent Algorithm for linear regression of large datasets
    P. Biedenkopf - 11.12.2020

"""

import gradient_decent as gd
import numpy as np


# Fish market dataset
dataset = './datasets/fish_market.csv'
data = np.genfromtxt(dataset, delimiter=',', skip_header=1)
y = data[:,1]
X = np.insert(data[:,2:], 0, data[:,0], axis=1)

# House prices dataset
# dataset = './datasets/house_prices.csv'
# data = np.genfromtxt(dataset, delimiter=',', skip_header=1)
# y = data[:,2]
# X = data[:,:2]


# Create algorithm instance
model = gd.GradienDescent(maxIter=300, alpha=0.1)

# Fit model with training data
model.train(X, y)


# Predict house price for example with 2000 sqf and 3 bedrooms
# X_predict = np.array([2000, 3])
# predicted_y = model.predict(X_predict)

# Predict the weight of a bream
X_predict = np.array([1, 23.4, 26.2, 30.6, 11.2, 4.5])
predicted_y = model.predict(X_predict)

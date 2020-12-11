""" 
    Gradient Descent Algorithm for linear regression of large datasets
    P. Biedenkopf - 11.12.2020

"""


import gradient_decent as gd
import numpy as np

# read training data
data = np.genfromtxt('data.txt', delimiter=',')
y = data[:,2]
X = data[:,0:2]

# Create algorithm instance
model = gd.GradienDescent(maxIter=200, alpha=0.1)

# Fit model with training data
model.train(X, y)

# Predict house price for example with 2000 sqf and 3 bedrooms
X_predict = [2000, 3]
predicted_y = model.predict(X_predict)


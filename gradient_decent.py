""" 
    Gradient Descent Algorithm for linear regression of large datasets
    P. Biedenkopf - 11.12.2020

"""

import numpy as np
import matplotlib.pyplot as plt

class GradienDescent:
    """ 
        member variables: 
            alpha   -   Learning rate
            maxIter -   Max iteration number
            J_hist  -   Objective function history list
    """
    def __init__(self, alpha=0.01, maxIter=500):
        self.alpha = alpha
        self.maxIter = maxIter
        self.J_hist = []
        
    def normalize(self, X):
        """ 
            normalizes the features of the problem
            variables: 
                alpha   - Learning rate
                maxIter - Max iteration number
        """
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)
        
        Xnorm = np.zeros(X.shape)
        for i in range(len(self.mu)):
            Xnorm[:,i] = (X[:,i]-self.mu[i]) / self.sigma[i]
            
        # add ones-vector from the left for constant features
        Xnorm = np.insert(Xnorm, 0, np.ones(X.shape[0]), axis=1) 
        return Xnorm

        
    def train(self, X, y, verbose=False):
        """ 
            Trains the model for a given dataset.
            variables: 
                X       -   dataset for which the model is trained
                y       -   labels of dataset
                verbose -   bool for detailed output
        """
        Xnorm = self.normalize(X)
        
        # initialize theta vector with zeros
        self.theta = np.zeros(Xnorm.shape[1])
        
        # main loop
        for i in range(self.maxIter):
            """ all steps need to be calculated before updating theta """
            steps = np.zeros(len(self.theta))
            for s in range(len(self.theta)):
                steps[s] = sum( (Xnorm.dot(self.theta) - y) * Xnorm[:,s] )
            
            self.theta = self.theta - self.alpha/Xnorm.shape[0] * steps
            costVal = costFunction(Xnorm, y, self.theta)
            
            if verbose:
                print("Iter: {} | Objective: {:e} | Theta: {}".format(i, costVal, self.theta))
            # add objective value to history list
            self.J_hist.append(costVal)
            
        self.plotConvergence()
        
    def predict(self, x):
        """ 
            Predicts value for a given dataset x.
            variables: 
                X - dataset for which a value gets predicted by the model
        """
        x_buffer = list(x) # for final print statment
        for i in range(len(self.sigma)):
            x[i] = (x[i] - self.mu[i]) / self.sigma[i]
        
        # add ones-vector from the left for constant features
        x = np.insert(x, 0, 1) 
            
        prediction = x.dot(self.theta)
        print("Predicted value for {} is: {:.2f}".format(x_buffer, prediction))
        return prediction
    
    def plotConvergence(self):
        plt.figure(1)
        plt.plot(range(len(self.J_hist)), self.J_hist, label="Gradient Descent ($\\alpha$ = {})".format(self.alpha))
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.title("Convergence")
        plt.legend()
    
    def __str__(self):
        return "Gradient Decent Algorithm with learning rate: {} and max. {} Iterations.".format(self.alpha, self.maxIter)


def costFunction(X, y, theta):
    """
            Returns objective value for measuring fitness of model
            variables: 
                X       - dataset for which a value gets predicted by the model
                y       - labels of dataset (vector)
                theta   - model parameters
        """
    y_predict = X.dot(theta)
    sqrtErrors = np.power((y_predict - y), 2)
    return 1/(2*len(y)) * sum(sqrtErrors)


    
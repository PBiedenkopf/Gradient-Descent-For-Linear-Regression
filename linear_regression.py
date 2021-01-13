""" 
    Gradient Descent Algorithm for linear regression of large datasets
    P. Biedenkopf - 11.12.2020

"""

import numpy as np
import matplotlib.pyplot as plt



class Regression:
    """ 
        Base class for the different regression types
    """
    def __init__(self, alpha=0.01, maxIter=500, verbose=False):
        self.alpha = alpha
        self.maxIter = maxIter
        self.verbose = verbose
        self.J_hist = []
        
    def normalize(self, X):
        """ 
            normalizes the features of the model
            variables: 
                alpha   - Learning rate
                maxIter - Max iteration number
        """
        self.mu = np.mean(X, axis=0)
        self.sigma = np.std(X, axis=0)
        
        # Normalize features
        Xnorm = np.zeros(X.shape)
        for i in range(len(self.mu)):
            Xnorm[:,i] = (X[:,i]-self.mu[i]) / self.sigma[i]
             
        return Xnorm
    
    def batchGradientDescent(self, X, y, costFun):
        """
                Performs batch gradient descent optimization
                variables: 
                    X       - dataset for which a value gets predicted by the model
                    y       - labels of dataset
                    costFun - Selected cost function
        """
        # initialize theta vector with zeros
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.maxIter):
            costVal, grad = costFun(X, y, self.theta)
            self.theta = self.theta - self.alpha/X.shape[0] * grad
            
            if self.verbose:
                print("Iter: {} | Objective: {:e} | Theta: {}".format(i, costVal, self.theta))
            
            # add objective value to history list
            self.J_hist.append(costVal)
            
        self.plotConvergence()

    def plotConvergence(self):
        plt.figure(1)
        plt.plot(range(len(self.J_hist)), self.J_hist, label="Gradient Descent ($\\alpha$ = {})".format(self.alpha))
        plt.grid(True), plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.title("Convergence")
        


class LinearRegression(Regression):
    """ 
        member variables: 
            alpha   -   Learning rate
            maxIter -   Max iteration number
            J_hist  -   Objective function history list
    """
    def __init__(self, alpha=0.1, maxIter=300):
        super().__init__(alpha, maxIter)
        
    def train(self, X, y):
        """ 
            Trains the model for a given dataset.
            variables: 
                X       -   dataset for which the model is trained
                y       -   labels of dataset
                verbose -   bool for detailed output
        """
        # Compute normalization
        Xnorm = self.normalize(X)
        
        # add ones-vector from the left for constant features
        Xnorm = np.insert(Xnorm, 0, np.ones(X.shape[0]), axis=1)
        
        # Minimize cost function with batch gradient descent
        self.batchGradientDescent(Xnorm, y, self.costFunction)
        
    def predict(self, x):
        """ 
            Predicts value for a given dataset x.
            variables: 
                X - dataset for which a value gets predicted by the model
        """
        # Normalize feature
        for i in range(len(self.sigma)):
            x[i] = (x[i] - self.mu[i]) / self.sigma[i]
        
        # add ones-vector from the left for constant features
        x = np.insert(x, 0, 1) 
            
        prediction = x.dot(self.theta)
        print("Predicted value: {:.2f}\n".format(prediction))
        return prediction
    

    def costFunction(self, X, y, theta):
        """
                Returns objective value and gradients regression model
                variables: 
                    X       - dataset for which a value gets predicted by the model
                    y       - labels of dataset
                    theta   - model parameters
                    grad    - gradients of objective (needed by gradient descent)
        """
        m = len(y)
        
        y_predict = X.dot(theta)
        sqrtErrors = np.power((y_predict - y), 2)
        J = 1/(2*m) * sum(sqrtErrors)
        
        grad = np.zeros(len(theta))
        
        for s in range(0, len(theta)):
            grad[s] = sum( (X.dot(theta) - y) * X[:,s] ) 
        
        return J, grad
    


class PolynomialRegression(Regression):
    """ 
        Regression with polynomial features and regularisation
    """
    def __init__(self, alpha=0.3, maxIter=2000, Lambda=5, power=5):
        super().__init__(alpha, maxIter)
        self.Lambda = Lambda
        self.power = power
        
    def train(self, X, y):
        """ 
            Trains the model for a given dataset.
            variables: 
                X       -   dataset for which the model is trained
                y       -   labels of dataset
                verbose -   bool for detailed output
        """
        # Add polynomial features
        Xpoly = self.addPolyFeatures(X, self.power)
        
        # Normalize features
        Xnorm = self.normalize(Xpoly)
        
        # add ones-vector from the left for constant features
        Xnorm = np.insert(Xnorm, 0, np.ones(X.shape[0]), axis=1)
        
        # Minimize regularized cost function with batch gradient descent
        self.batchGradientDescent(Xnorm, y, self.regCostFunction)
        
        
    def predict(self, x):
        """ 
            Predicts value for a given dataset x.
            variables: 
                X - dataset for which a value gets predicted by the model
        """
        # Add the polynomial features
        x = self.addPolyFeatures(x, self.power)
        
        
        # Normalize the features
        for i in range(len(self.sigma)):
            x[i] = (x[i] - self.mu[i]) / self.sigma[i]
        
        # add ones for bias term
        x = np.insert(x, 0, 1)

        prediction = x.dot(self.theta)
        print("Predicted value: {:.2f}\n".format(prediction))
        return prediction

        
    def addPolyFeatures(self, X, p):
        """
            Adds polynomial features upto the p-th power the model
            variables: 
                X       - dataset for which a value gets predicted by the model
                p       - Power for polynomial features
        """
        # getting the feature size can cause a problem when shape of prediction is (x,) instead of (x,1)
        try:
            featureSize = X.shape[1]
            Xpoly = X
            for deg in range(2, p+1):
                for i in range(1, featureSize):
                    Xpoly = np.hstack( ( Xpoly, np.array([ np.power(Xpoly[:,i],deg) ]).T ) )
        except:
            featureSize = len(X)
            Xpoly = X
            for deg in range(2, p+1):
                for i in range(1, featureSize):
                    Xpoly = np.append( Xpoly, np.power(Xpoly[i], deg) )
                    
        return Xpoly
    
        
    def regCostFunction(self, X, y, theta):
        """
            Regularized cost function for regressions with polynomial features
            Returns objective value and gradients regression model
            variables: 
                X       - dataset for which a value gets predicted by the model
                y       - labels of dataset
                theta   - model parameters
                grad    - gradients of objective (needed by gradient descent)
        """
        m = len(y)
        y_predict = X.dot(theta)
        sqrtErrors = np.power((y_predict - y), 2)
        regul = self.Lambda/(2*m) * sum( np.power(theta[1:], 2) )
        J = 1/(2*m) * sum(sqrtErrors) + regul
        
        grad = np.zeros(len(theta))
        
        # Exclude bias term from regularized
        grad[0] = 1/m * sum( (X.dot(theta) - y) * X[:,0] )
        
        for s in range(1, len(theta)):
            regul_grad = self.Lambda / m * theta[s]
            grad[s] = 1/m * sum( (X.dot(theta) - y) * X[:,s] ) + regul_grad
        
        return J, grad
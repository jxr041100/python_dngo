import numpy as np
import statsmodels.api as sm
import sklearn.linear_model as sklm
import matplotlib.pyplot as plt
from numpy import atleast_2d as vec

class LinearRegressor():

    def __init__(self, dataset, intercept=True):
        """Initialization of Optimizer object
        
        Keyword arguments:
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__dataset = dataset
        self.__intercept = intercept

    def train(self):
        dataset = self.__dataset
        intercept = self.__intercept
        train_X = sm.add_constant(dataset[:, :-1]) if intercept else dataset[:, :-1]
        train_Y = dataset[:, -1:]

        XX_inv,_,_,_ = np.linalg.lstsq(np.dot(train_X.T, train_X), 
                                       np.identity(train_X.shape[1])) 
        beta = np.dot(np.dot(XX_inv, train_X.T), train_Y)
        
        self.__XX_inv = XX_inv
        self.__beta = beta
        
    def predict(self, test_X):
        dataset = self.__dataset
        intercept = self.__intercept
        XX_inv  = self.__XX_inv
        beta    = self.__beta
        
        train_X = sm.add_constant(dataset[:, :-1]) if intercept else dataset[:, :-1]
        test_X = sm.add_constant(vec(test_X)) if intercept else vec(test_X)
        train_Y = dataset[:, -1:]
        train_pred = np.dot(train_X, beta)
        
        # Confidence interval
        sig = (np.linalg.norm(train_Y-train_pred)**2/(train_X.shape[0]-train_X.shape[1]+1))**0.5
        s = []
        for row in range(test_X.shape[0]):
            x = test_X[[row], :]
            s.append(sig*(1 + np.dot(np.dot(x, XX_inv), x.T))**0.5)
            
        s = np.reshape(np.asarray(s), (test_X.shape[0], 1))

        test_pred = np.dot(test_X, beta)
        hi_ci = test_pred + 2*s
        lo_ci = test_pred - 2*s

        return test_pred, hi_ci, lo_ci

    def predict_reg(self, test_X):
        clf = sklm.Lasso(alpha=1, fit_intercept=False)
        clf.fit(self.__dataset[:, :-1], self.__dataset[:, -1:])
        print np.atleast_2d(clf.coef_)
        pred =clf.predict(test_X)
        pred = np.atleast_2d(pred).T
        return pred, pred, pred     


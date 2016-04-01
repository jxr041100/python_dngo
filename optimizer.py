import time
import numpy as np
import neural_net as nn
import linear_regressor as lm
import scipy.stats as stats
import matplotlib.pyplot as plt
import random
from hidden_function import evaluate
import statsmodels.api as sm

class Optimizer(object):

    def __init__(self, dataset, domain):
        """Initialization of Optimizer object
        
        Keyword arguments:
        dataset -- an n by (m+1) array that forms the matrix [X, Y]
        """
        self.__dataset = dataset
        nobs = dataset.shape[0]
        self.__architecture = (domain.shape[1], 50, 50, nobs - 2 if nobs < 50 else 50, 1 )
        self.__domain = domain

    def train(self):
        """ Using the stored dataset and architecture, trains the neural net to 
        perform feature extraction, and the linear regressor to perform prediction
        and confidence interval computation.
        """
        neural_net = nn.NeuralNet(self.__architecture, self.__dataset)
        neural_net.train()
        self.__W, self.__B = neural_net.extract_params()
        self.__nn_pred = neural_net.e.network.predict(self.__domain)

        # Extract features
        train_X = self.__dataset[:, :-1]
        train_Y = self.__dataset[:, -1:]
        train_features = self.extract_features(train_X)
        domain_features = self.extract_features(self.__domain)
        lm_dataset = np.concatenate((train_features, train_Y), axis=1)

        # Train and predict with linear_regressor
        linear_regressor = lm.LinearRegressor(lm_dataset, intercept=False)
        linear_regressor.train()
        self.__pred, self.__hi_ci, self.__lo_ci = linear_regressor.predict(domain_features)

    def retrain_NN(self):
        neural_net = nn.NeuralNet(self.__architecture, self.__dataset)
        neural_net.train()
        self.__W, self.__B = neural_net.extract_params()

    def retrain_LR(self):
        """ After the selected point (see select()) is queried, insert the new info
        into dataset. Depending on the size of the dataset, the module decides whether
        to re-train the neural net (for feature extraction). 
        A new interpolation is then constructed.

        Keyword arguments:
        new_data -- a 1 by (m+1) array that forms the matrix [X, Y]
        """

        train_X = self.__dataset[:, :-1]
        train_Y = self.__dataset[:, -1:]

        # Extract features
        train_features = self.extract_features(train_X)
        domain_features = self.extract_features(self.__domain)
        lm_dataset = np.concatenate((train_features, train_Y), axis=1)

        # Train and predict with linear_regressor
        linear_regressor = lm.LinearRegressor(lm_dataset)
        linear_regressor.train()
        self.__pred, self.__hi_ci, self.__lo_ci = linear_regressor.predict(domain_features)

    def extract_features(self, test_X):
        W = self.__W
        B = self.__B
        architecture = self.__architecture

        # Feedforward into custom neural net
        X = []
        for i in range(test_X.shape[0]):
            test_val = test_X[[i], :]
            L = np.tanh(np.dot(test_val, W[0]) + B[0])
            
            for i in range(1, len(architecture)-2):
                L = np.tanh(np.dot(L, W[i]) + B[i])
                
            X.extend(L.tolist())
                
        X = np.asarray(X)
        X = sm.add_constant(X)

        return X

    def select_multiple(self, cap=5):
        """ Identify multiple points. 
        """
        
        # Rank order by expected improvement
        train_Y    = self.__dataset[:, -1:]
        prediction = self.__pred
        hi_ci      = self.__hi_ci

        sig = abs((hi_ci - prediction)/2)
        gamma = (prediction - np.max(train_Y)) / sig
        ei = sig*(gamma*stats.norm.cdf(gamma) + stats.norm.pdf(gamma))

        if np.max(ei) <= 0:
            # If no good points, do pure exploration
            sig_order = np.argsort(-sig, axis=0)
            select_indices = sig_order[:cap, 0].tolist()
        else:
            ei_order = np.argsort(-1*ei, axis=0)
            select_indices = [ei_order[0, 0]]

            for candidate in ei_order[:, 0]:
                keep = True
                for selected_index in select_indices:
                    keep = keep*self.check_point(selected_index, candidate)
                if keep and ei[candidate, 0] > 0:
                    select_indices.append(candidate)
                if len(select_indices) == cap: # Number of points to select
                    break 

            if len(select_indices) < cap:
                # If not enough good points, append with exploration
                sig_order = np.argsort(-sig, axis=0)
                add_indices = sig_order[:(cap-len(select_indices)), 0].tolist()
                select_indices.extend(add_indices)

        index = np.argmax(ei)
        self.__gamma = gamma
        self.__ei = ei

        return np.atleast_2d(self.__domain[select_indices, :])

    def check_point(self, selected_index, order):
        prediction = self.__pred
        hi_ci      = self.__hi_ci

        sig = (hi_ci[selected_index] - prediction[selected_index])/2
        z_score = abs(prediction[order] - prediction[selected_index])/sig

        return (stats.norm.cdf(-z_score)*2) < 0.5

    def update_data(self, new_data):
        nobs = self.__dataset.shape[0]
        if nobs < 50:
            nobs += new_data.shape[0]
            self.__architecture = (self.__domain.shape[1], 50, 50, nobs - 2 if nobs < 50 else 50, 1)
        
        self.__dataset = np.concatenate((self.__dataset, new_data), axis=0)
    
    def update_params(self, W, B, architecture):
        self.__W = W
        self.__B = B
        self.__architecture = architecture

    def get_prediction(self):
        return (self.__domain, self.__pred, self.__hi_ci, 
                self.__lo_ci, self.__nn_pred, self.__ei, self.__gamma)

    def get_dataset(self):
        return self.__dataset
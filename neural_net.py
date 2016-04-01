import numpy as np
import theanets
import statsmodels.api as sm

class NeuralNet(object):
    
    def __init__(self, architecture, dataset):
        self.__architecture = architecture
        self.__dataset = dataset
        self.e = None
    
    def train(self):
        architecture = self.__architecture
        dataset = self.__dataset

        cut = int(0.9 * len(dataset))  # select 90% of data for training, 10% for validation
        idx = range(len(dataset))
        np.random.shuffle(idx)
        
        train = idx[:cut]
        train_set = [dataset[train, :-1], dataset[train, -1:]]
        valid = idx[cut:]
        valid_set = [dataset[valid, :-1], dataset[valid, -1:]]
        
        e = theanets.Experiment(theanets.feedforward.Regressor,
                                layers=architecture)
        
        e.train(train_set, valid_set,optimize='sgd',hidden_activation='tanh',output_activation='linear',learning_rate=0.01)
        self.e = e

    def extract_params(self):
        architecture = self.__architecture
        e = self.e
        # Extract parameters
        W = {}
        B = {}
        for i in range(len(architecture)-2):
            W[i] = e.network.params[2*i].get_value()
            B[i] = np.reshape(e.network.params[2*i+1].get_value(), (1, architecture[i+1]))
            
        self.__W = W
        self.__B = B
        
        return (W, B)


"""
@Author: Jianfeng Ren
@Date: 4/01/2016

Performs scale bayesian optimization using the deep neural network.
"""
from   hidden_function import evaluate,get_settings
import matplotlib.pyplot as plt
import optimizer as op
import numpy as np

print_statements = True

# Get settings related with hypter parameters space. 
# domain can be image features for image domain, or be text features in text domain
lim_domain, init_size, additional_query_size, init_query, domain, selection_size = get_settings()

# Construct the dataset
dataset = evaluate(init_query[0,:], lim_domain)

print "Randomly query a set of initial points... ",

for query in init_query[1:,:]:
    dataset = np.concatenate((dataset, evaluate(query, lim_domain)), axis=0)

print "Complete initial dataset acquired"
    
# Begin sequential optimization using NN-LR based query system
optimizer = op.Optimizer(dataset, domain)
optimizer.train()

# Select a series of points to query
selected_points = optimizer.select_multiple(selection_size) # (#points, m) array
selection_index = 0

print "Performing optimization..."

for i in range(additional_query_size):
    if selection_index == selection_size:
        # Update optimizer's dataset and retrain LR
        optimizer.retrain_LR()                            
        selected_points = optimizer.select_multiple(selection_size) # Select new points
        selection_size = selected_points.shape[0]     # Get number of selected points
        selection_index = 0                           # Restart index      

    if (optimizer.get_dataset().shape[0] % 100) == 0:
        # Retrain the neural network
        optimizer.retrain_NN()

    new_data = evaluate(selected_points[selection_index], lim_domain)
    optimizer.update_data(new_data)
    selection_index += 1

    if (i+1) % (additional_query_size/10) == 0:
            print "%.3f completion..." % ((i+1.)/additional_query_size)

print "Sequential optimization task complete."
print "Best evaluated point is:"
dataset = optimizer.get_dataset()
print dataset[np.argmax(dataset[:, -1]), :]
print "Predicted best point is:"
optimizer.retrain_LR()
domain, pred, hi_ci, lo_ci, nn_pred, ei, gamma = optimizer.get_prediction()
index = np.argmax(pred[:, 0])
print np.concatenate((np.atleast_2d(domain[index, :]), np.atleast_2d(pred[index, 0])), axis=1)[0, :]

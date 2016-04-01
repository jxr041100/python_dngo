import numpy as np
import time
from gaussian_process import gaussian_process as gp

def get_settings(lim_domain_only=False):   
    lim_domain = np.array([[-1.],[ 1.]])

    if lim_domain_only:
        return lim_domain

    init_size = 50
    additional_query_size = 300
    selection_size = 5

    # Get initial set of locations to query
    init_query = np.random.uniform(0, 1, size=(init_size, lim_domain.shape[1]))

    # Establish the grid size to use. 
    domain = np.atleast_2d(np.linspace(-1, 1, 5000)).T   

    return lim_domain, init_size, additional_query_size, init_query, domain, selection_size

def evaluate(query, lim_domain):
    var     = (lim_domain[1, :] - lim_domain[0, :])/2.
    mean    = (lim_domain[1, :] + lim_domain[0, :])/2.
    query   = np.atleast_2d(query)      # Convert to (1, m) array
    X       = query*var + mean          # Scale query to true input space
    dataset = np.concatenate((query, gp(X) + np.random.randn()/100), axis=1)    
    return dataset
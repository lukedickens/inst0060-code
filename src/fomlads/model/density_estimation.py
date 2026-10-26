"""
Provided code version for tutorial 04
"""
import numpy as np

def max_lik_mv_gaussian(data):
    """
    Finds the maximum likelihood mean and covariance matrix for gaussian data
    samples (data)

    parameters
    ----------
    data - data array, 2d array of samples, each row is assumed to be an
      independent sample from a multi-variate gaussian

    returns
    -------
    mu - mean vector
    Sigma - 2d array corresponding to the covariance matrix  
    """
    # TODO: edit this function to calculate the maximum likelihood parameters
    N, dim = data.shape
    mu = np.zeros(dim)
    Sigma = np.identity(dim)
    return mu, Sigma


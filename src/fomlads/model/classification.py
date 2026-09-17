import csv
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from fomlads.data.function import logistic_sigmoid
from fomlads.model.density_estimation import max_lik_mv_gaussian

def project_data(data, weights):
    """
    Projects data onto single dimension according to some weight vector

    parameters
    ----------
    data - a 2d data matrix (shape NxD array-like)
    weights -- a 1d weight vector (shape D array like)

    returns
    -------
    projected_data -- 1d vector (shape N np.array)
    """
    # TODO: you should correct this code so that it projects
    # TODO: the data down onto 1 dimension using the weight
    # TODO: vector.
    # NOTE: Currently it just returns a random set of values
    return np.random.random(data.shape[0])

def fisher_linear_discriminant_projection(inputs, targets):
    """
    Finds the direction of best projection based on Fisher's linear discriminant

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1

    returns
    -------
    weights - a normalised projection vector corresponding to Fisher's linear 
        discriminant
    """
    # TODO: write this method as described on the exercise sheet.
    return None

def maximum_separation_projection(inputs, targets):
    """
    Finds the projection vector that maximises the distance between the 
    projected means

    parameters
    ----------
    inputs - a 2d input matrix (array-like), each row is a data-point
    targets - 1d target vector (array-like) -- can be at most 2 classes ids
        0 and 1

    returns
    -------
    weights - a normalised projection vector
    """
    # get the shape of the data
    N, D = inputs.shape
    # separate the classes
    inputs0 = inputs[targets==0]
    inputs1 = inputs[targets==1]
    # find maximum likelihood approximations to the two data-sets
    m0,_ = max_lik_mv_gaussian(inputs0)
    m1,_ = max_lik_mv_gaussian(inputs1)
    # calculate weights vector
    weights = m1-m0
    return weights


import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

# this is the feature mapping for a polynomial of given degree in 1d
def expand_to_monomials(inputs, degree):
    """
    Create a design matrix from a 1d array of input values, where columns
    of the output are powers of the inputs from 0 to degree (inclusive)

    So if input is: inputs=np.array([x1, x2, x3])  and degree = 4 then
    output will be design matrix:
        np.array( [[  1.    x1**1   x1**2   x1**3   x1**4   ]
                   [  1.    x2**1   x2**2   x2**3   x2**4   ]
                   [  1.    x3**1   x3**2   x3**3   x3**4   ]])
    """
    expanded_inputs = []
    for i in range(degree+1):
        expanded_inputs.append(inputs**i)
    return np.array(expanded_inputs).transpose()


def least_squares_weights(inputs, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.
    """
    # renaming for readability
    Phi = inputs
    targets = targets.reshape((-1,1))
    weights = linalg.inv(Phi.T @ Phi) @ Phi.T @ targets
    return weights.flatten()


# two names for the same function
ml_weights = least_squares_weights

def regularised_least_squares_weights(
        inputs, targets, lambda_):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets penalised by some regularisation term
    (lambda_)
    """
    # renaming for readability
    Phi = inputs
    targets = targets.reshape((-1,1))
    I = np.identity(Phi.shape[1])
    weights = linalg.inv(lambda_*I + Phi.T @ Phi) @ Phi.T @ targets
    return weights.flatten()

# two names for the same function
regularised_ml_weights = regularised_least_squares_weights

def construct_polynomial(degree, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights. It returns a polynomial of the univariate input x
    # if given an array of xs, then it gives one functional value per input
    def polynomial_function(xs):
        monomials_mtx = expand_to_monomials(xs, degree)
        ys = monomials_mtx @ weights.reshape((-1,1))
        return ys.flatten()
    # we return the function reference (handle) itself. This can be used like
    # any other function
    return polynomial_function



import csv
import numpy as np
import numpy.random as random
import numpy.linalg as linalg
import matplotlib.pyplot as plt

from poly_fit_plot import plot_function
from poly_fit_plot import plot_function_and_data
from poly_fit_plot import plot_function_data_and_approximation

def simple_sin(inputs):
    return np.sin(2*np.pi*inputs)

def expand_to_monomials(inputs, degree):
    # create a list of the all inputs raise to each possible power
    expanded_inputs = []
    for i in range(degree+1):
        expanded_inputs.append(inputs**i)
    return np.array(expanded_inputs).T

def construct_polynomial(degree, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        expanded_xs = np.matrix(expand_to_monomials(xs, degree))
        ys = expanded_xs*np.matrix(weights).reshape((len(weights),1))
        return np.array(ys).flatten()
    # we return the function reference (handle) itself. This can be used like
    # any other function
    return prediction_function

def sample_data(N, true_func, include_xlim=True):
    """
    Sample input and target data for regression. Produces random inputs between
    0 and 1 and noise corrupted outputs of the true function.

    Parameters
    ----------
    N - the number of data points to output
    true_func - the true underlying function 
    include_xlim (optional) - whether to include 0 and 1 in the inputs (this can
      improve the stability of your regression results.

    Returns
    -------
    inputs - randomly sampled input data (x)
    targets - the associated targets (true_func(x) + gaussian noise)
    """
    # inputs are a collection of N random numbers between 0 and 1
    # for stability we include points at 0 and 1.
    inputs = random.uniform(size=N)
    if include_xlim and N >2:
        inputs[0] = 0
        inputs[-1] = 1
    # outputs are sin(2*pi*x) + gaussian noise
    targets = true_func(inputs) + random.normal(loc=0.0, scale=0.3, size=N)
    return inputs, targets

def sample_data_dispersed_inputs(N, function):
    """
    The uniform distribution can lead to clumps of data which in turn do not
    give very stable samples for fitting. This method produces sample points
    that are a little more evenly distributed through space. The underlying
    distribution is similar to (but not the same as) a poisson process.
    """
    # inputs are a collection of N random numbers between 0 and 1
    # generate a random sequence of interpoint intervals
    inputs = sample_dispersed_inputs(N)
    print("inputs = %r" % (inputs,))
    # outputs are sin(2*pi*x) + gaussian noise
    targets = function(inputs) + random.normal(loc=0.0, scale=0.3, size=N)
    print("targets = %r" % (targets,))
    return inputs, targets

def sample_dispersed_inputs(N):
    #waits = random.exponential(size=N)
    waits = random.gamma(shape=2., scale=2, size=N)
    # convert to increasing sequence of N+1 point positions (unnormalised)
    # first point at 0
    accum = 0
    raw = [ accum ]
    for wait in waits:
        accum += wait
        raw.append(accum)
    # normalise so last point at 1
    raw = np.array(raw)/raw[-1]
    # randomly bisect last interval and add to all positions 1 to N (not N+1)
    shift = random.uniform()
    inputs = raw[:-1] + shift*(raw[-1] - raw[-2])
    return inputs

def least_squares_weights(Phi, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs Phi and the targets.
    
    parameters
    ----------
    Phi - a 2d array of inputs, where each row of Phi is a vector representation
        of data-point i.
    targets - a 1d array of targets
    
    returns
    -------
    weights - a 1d array of best fit weights for the model
    """
    targets = targets.reshape((-1,1))
    weights = linalg.inv(Phi.T @ Phi) @ Phi.T @ targets
    return weights.flatten()

def regularised_least_squares_weights(
        Phi, targets, lambda_):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets penalised by some regularisation term
    (lambda_)

    parameters
    ----------
    Phi - a 2d array of inputs, where each row of Phi is a vector representation
        of data-point i.
    targets - a 1d array of targets
    
    returns
    -------
    weights - a 1d array of best fit weights for the model
    """
    targets = targets.reshape((-1,1))
    I = np.identity(Phi.shape[1])
    weights = linalg.inv(lambda_*I + Phi.T @ Phi) @ Phi.T @ targets
    return weights.flatten()

def test_functionality():
    """
    This function contains example code that demonstrates how to use the 
    functions defined in poly_fit_base for fitting polynomial curves to data.
    """
    # some dummy inputs
    dummy_inputs = np.array([0.9,1,1.1])
    # choose the degree of polynomial for your family of approximating functions
    degree = 9
    # now plot the feature vector for three dummy inputs to see what it does
    expanded_dummy_inputs = expand_to_monomials(dummy_inputs, degree)
    print("expanded_dummy_inputs = %r" % (expanded_dummy_inputs,))
    plt.plot(
      np.arange(degree+1), expanded_dummy_inputs.T, 's',
      markersize=3)


    # choose number of data-points and sample a pair of vectors: the input
    # values and the corresponding target values
    N = 10
    inputs, targets = sample_data(N, simple_sin)
    # now plot this data
    plot_function_and_data(inputs, targets, simple_sin)
    # convert our inputs (we just sampled) into a matrix where each row
    # is a vector of monomials of the corresponding input
    processed_inputs = expand_to_monomials(inputs, degree)
    #
    # find the weights that fit the data in a least squares way
    weights = least_squares_weights(processed_inputs, targets)
    print("weights = %r" % (weights,))
    # use weights to create a function that takes inputs and returns predictions
    # in python, functions can be passed just like any other object
    # those who know MATLAB might call this a function handle
    trained_func = construct_polynomial(degree, weights)
    plot_function_data_and_approximation(
        trained_func, inputs, targets, simple_sin)
    #
    # fit data with regularised weights and then plot 
    lambda_ = np.exp(-18)
    # a new set of weights is found which gives a new prediction function
    weights2 = regularised_least_squares_weights(
        processed_inputs, targets,  lambda_)
    trained_func2 = construct_polynomial(degree, weights2)
    plot_function_data_and_approximation(
        trained_func2, inputs, targets, simple_sin)
    #
    plt.show()

if __name__ == '__main__':
    # this bit only runs when this script is called from the command line
    # but not when poly_fit_base.py is used as a library
    test_functionality()

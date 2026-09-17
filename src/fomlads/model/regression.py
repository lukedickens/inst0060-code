import csv
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

# to improve the coherence of modules expand_to_monomials
# has been moved to the basis_functions module
from fomlads.model.basis_functions import expand_to_monomials

def linear_model_predict(inputs, weights):
    """
    Returns predictions for linear model

    parameters
    ----------
    inputs - a 2d array (shape NxD) representing the data matrix, X, or
      design matrix, Phi, of  data-points from regression dataset. Each row
      inputs[i,:] is the feature vector of the ith data-point and each column
      inputs[:,j] is the ordered list of values for the jth feature. 
    weights - the weights of your model as a 1d numpy array
    """
    ## TODO: you must implement the linear_model_predict function
    ## TODO: currently this just returns a dummy output
    return np.zeros(inputs.shape[0])


def ml_weights(inputs, targets):
    """
    This method returns the weights that give the best linear fit between
    the processed inputs and the targets.

    parameters
    ----------
    inputs - a 2d array (shape NxD) representing the data matrix, X, or
      design matrix, Phi, of  data-points from regression dataset. Each row
      inputs[i,:] is the feature vector of the ith data-point and each column
      inputs[:,j] is the ordered list of values for the jth feature. 
    targets - a 1d array (size N) the associated targets from regression dataset

    returns
    -------
    weights - a 1d array (size D) of maximum likelihood weights
    """
## TODO: you must implement the ml_weights function here
## TODO: currently this just returns a dummy output.
    return np.zeros(inputs.shape[1])

## TODO: you must add the regularised_ml_weights function
## TODO: the input arguments are:
## TODO:     designmtx - the design matrix as a 2d numpy array 
## TODO:     targets - the real valued targets of your data as a 1d numpy array
## TODO:     lambda_ - the real valued regularisation parameter
## TODO: use the ml_weights function as a starting point

def construct_polynomial(degree, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights.
    def polynomial_function(xs):
        """
        A univariate polynomial prediction function
        
        parameters
        ----------
        xs - 1d array of inputs size N

        returns
        -------
        predictions - a 1d array of predictions y, where y[i] is the prediction
            for data point xs[i,:]
        """
        monomials_mtx = expand_to_monomials(xs, degree)
        ys = monomials_mtx @ weights.reshape((-1,1))
        return ys.flatten()
    # we return the function itself as a variable. This can be used like
    # any other function
    return polynomial_function

def construct_feature_mapping_function(feature_mapping, weights):
    """
    This function creates and returns a prediction function based on a
    feature mapping and some weights.

    The returned prediction function takes a set of input values and returns
    the predicted output for each.
    """
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(X):
        """
        A linear model prediction function
        
        parameters
        ----------
        X - 2d (M,D)-array of inputs, where M is the number of
            test data-points and D is the dimension of the points (rows)
            of inputs

        returns
        -------
        predictions - a 1d array of predictions y, where y[i] is the prediction
            for data point X[i,:]
        """
        designmtx = feature_mapping(X)
        return linear_model_predict(designmtx, weights)
    # we return the function itself as a variable. This can be used like
    # any other function
    return prediction_function

def construct_knn_function_1d(training_inputs, targets, k):
    """
    For 1 dimensional training data, it produces a function f:reals-> reals
    that outputs the mean training value in the k-Neighbourhood of any input.

    parameters
    ----------
    training_inputs - 1d array (size N) of  data-points from regression dataset
    targets - the associated targets from regression dataset
    k - the number of neighbours on which to base the prediction.

    returns
    -------
    prediction_function - a function that takes 1d array (size M) of test inputs 
      xs and outputs a 1d array of predictions ys, where ys[i] is the prediction
      for test input xs[i]
    """
    N = training_inputs.size
    # here is a function that is created on the fly from the input feature
    # mapping and weights
    def prediction_function(xs):
        """
        A univariate KNN prediction function
        
        parameters
        ----------
        xs - 1d array of inputs size N

        returns
        -------
        predictions - a 1d array of predictions y, where y[i] is the prediction
            for data point xs[i,:]
        """
        M = xs.size
        ## TODO: You must edit this function so that distances is a 2d array 
        ## TODO: where distances[i,j] contains the absolute difference between
        ## TODO: xs[i] and training_inputs[j] 
        distances = np.random.rand(M,N)
        predicts = np.empty(M)
        # each row of each_k_neighbours is the indices of the k
        # neighbours of test_input[i] in training_inputs
        each_k_neighbours = np.argpartition(distances, kth=k, axis=-1)[:,:k]
        for i, neighbourhood in enumerate(each_k_neighbours):
            # the neighbourhood is the indices of the closest training inputs 
            # to xs[i] the prediction is the mean of the targets
            # for this neighbourhood
            predicts[i] = np.mean(targets[neighbourhood])
        return predicts
    # We return the function as a variable
    return prediction_function

def construct_knn_function(training_inputs, targets, k, metric):
    """
    Produces a function with signature  f:R^D-> R  that outputs the mean
    training value in the k-Neighbourhood of any D dimensional input.
    
    parameters
    ----------
    training_inputs - 2d (N,D)-array of inputs, where N is the number of training
        data-points and D is the dimension of the points (rows) of inputs
    targets - the associated targets from regression dataset
    k - the number of neighbours on whic to base the prediction.
    metric - the distance function which takes 2 2d arrays as input, and 
        produces a matrix of distances between each point (row) in X with each
        point (row) in Y. For instance,

            distances = metric(X, Y) 

        is a valid call if X and Y are both 1d arrays of size (Nx,D) and (Ny,D)
        respectively. This call must produce an 2d output array of distances
        where distances[i,j] equals the distance between X[i,:] and Y[j,:].

    returns
    -------
    prediction_function - a function that takes 2d (M,D)-array of inputs X and 
        outputs a 1d array of predicitons y, where y[i] is the prediction for
        data point X[i,:]
    """
    def prediction_function(X):
        """
        A KNN prediction function
        
        parameters
        ----------
        X - 2d (M,D)-array of inputs, where N is the number of
            test data-points and D is the dimension of the points (rows)
            of inputs

        returns
        -------
        predictions - a 1d array of predictions y, where y[i] is the prediction
            for data point X[i,:]
        """
        M, D = X.shape
        distances = metric(X, training_inputs)
        predicts = np.empty(M)
        for i, neighbourhood in enumerate(np.argpartition(distances, k)[:,:k]):
            # the neighbourhood is the indices of the closest inputs to xs[i]
            # the prediction is the mean of the targets for this neighbourhood
            predicts[i] = np.mean(targets[neighbourhood])
        return predicts
    # We return a handle to the locally defined function
    return prediction_function



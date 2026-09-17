import numpy as np
import matplotlib.pyplot as plt

# for fitting
from fomlads.model.regression import least_squares_weights
from fomlads.model.regression import regularised_least_squares_weights

def root_mean_squared_error(y_true, y_pred):
    """
    Evaluate how closely predicted values (y_pred) match the true values
    (y_true, also known as targets)

    Parameters
    ----------
    y_true - the true targets
    y_pred - the predicted targets

    Returns
    -------
    mse - The root mean squared error between true and predicted target
    """
    # be careful, square must be done element-wise
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true.flatten() - y_pred.flatten())**2)
    return np.sqrt(mse)


## code to generate and apply partitions
def train_and_test_partition(inputs, targets, train_filter, test_filter):
    """
    Splits a data matrix (or design matrix) and associated targets into train
    and test parts.

    parameters
    ----------
    inputs - a 2d numpy array whose rows are the datapoints, or can be a design
        matric, where rows are the feature vectors for data points.
    targets - a 1d numpy array whose elements are the targets.
    train_filter - A list (or 1d array) of N booleans, where N is the number of
        data points. If the ith element is true then the ith data point will be
        added to the training data.
    test_filter - (like train_filter) but specifying the test points.

    returns
    -------     
    train_inputs - the training input matrix
    train_targets - the training targets
    test_inputs - the test input matrix
    test_targets - the test targtets
    """
    # get the indices of the train and test portion
    univariate = (len(inputs.shape) == 1)
    if univariate:
        # if inputs is a sequence of scalars we should reshape into a matrix
        inputs = inputs.reshape((inputs.size,1))
    train_inputs = inputs[train_filter,:]
    test_inputs = inputs[test_filter,:]
    train_targets = targets[train_filter]
    test_targets = targets[test_filter]
    if univariate:
        train_inputs = train_inputs.flatten()
        test_inputs = test_inputs.flatten()
    return train_inputs, train_targets, test_inputs, test_targets



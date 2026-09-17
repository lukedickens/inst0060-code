import numpy as np
import matplotlib.pyplot as plt

from poly_fit_base import least_squares_weights
from poly_fit_base import regularised_least_squares_weights
from poly_fit_base import simple_sin
from poly_fit_base import expand_to_monomials
from poly_fit_base import create_prediction_function
from poly_fit_base import sample_data

from poly_fit_plot import plot_train_test_errors

def train_and_test(
        degree, train_inputs, train_targets, test_inputs, test_targets,
        lambda_=None):
    """
    Fits a polynomial of degree "degree" to your training data then evaluates
    train and test errors and plots the resulting curves

    Parameters
    ----------
    degree - the degree of polynomial to fit
    train_inputs - the training inputs
    train_targets - the training targets
    test_inputs - the test inputs
    test_targets - the test targets
    lambda_ (optional) - the regularisation strength. If not provided then
        the non-regularised least squares method is used.

    Returns
    -------
    train_error - the training error for the approximation
    test_error - the test error for the approximation
    """
    # convert both train and test inputs to monomial vectors
    processed_train_inputs = expand_to_monomials(train_inputs, degree)
    processed_test_inputs = expand_to_monomials(test_inputs, degree)
    # find the weights, using least squares or regularised least squares
    if lambda_ is None:
        # use simple least squares approach
        weights = least_squares_weights(processed_train_inputs, train_targets)
    else:
        # use regularised least squares approach
        weights = regularised_least_squares_weights(
            processed_train_inputs, train_targets,  lambda_)
    # create the prediction function
    trained_func = create_prediction_function(degree, weights)
    # get the train and test errors and return them
    train_error = root_mean_squared_error(train_targets, trained_func(train_inputs))
    test_error = root_mean_squared_error(test_targets, trained_func(test_inputs))
    return train_error, test_error

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
    N = len(y_true)
    # be careful, square must be done element-wise (hence conversion
    # to np.array)
    mse = np.sum((np.array(y_true).flatten() - np.array(y_pred).flatten())**2)/N
    return np.sqrt(mse)

def evaluate_degree(lambda_, N, degree_sequence=[0,1,2,3,4,5,6,7,8,9]):
    """
    Evaluates and plots test & train error (RMSE) for different degrees of
    polynomial fit to synthetic data.
    Allows one to essentially recreate Figure 1.5 from Bishop.

    Parameters
    ----------
    lambda_ - the regularisation parameter if one is being used. Set to None
        if conventional least squares fit needed
    N - number of data points to sample for training
    degree_sequence - specifies which degrees of polynomial to fit to the data
    """
    # sample  train and test data
    train_inputs, train_targets = sample_data(N, simple_sin)
    test_inputs, test_targets = sample_data(100, simple_sin)
    # 
    train_errors = []
    test_errors = []
    for degree in degree_sequence:
        # for each degree fit the data then evaluated train/test error
        train_error, test_error = train_and_test(
            degree, train_inputs, train_targets, test_inputs,
            test_targets, lambda_)
        train_errors.append(train_error)
        test_errors.append(test_error)
    # plot the results
    plot_train_test_errors("degree", degree_sequence, train_errors, test_errors)
    plt.show()

def evaluate_regulariser(degree, N, lambdas=None):
    """
    Evaluates and plots test & train error (RMSE) for different regularisation
    strength.
    Allows one to essentially recreate Figure 1.8 from Bishop.

    Parameters
    ----------
    degree - the degree of polynomial to fit
    N - number of data points to sample for training
    lambdas - specifies which regularisation parameters to fit with
    """
    if lambdas is None:
        lambdas = np.logspace(-17, -9, 50)
    # sample  train and test data
    train_inputs, train_targets = sample_data(N, simple_sin)
    test_inputs, test_targets = sample_data(100, simple_sin)
    # 
    train_errors = []
    test_errors = []
    for lambda_ in lambdas:
        # for each degree fit the data then evaluated train/test error
        train_error, test_error = train_and_test(
            degree, train_inputs, train_targets, test_inputs,
            test_targets, lambda_)
        train_errors.append(train_error)
        test_errors.append(test_error)
    # plot the results
    fig, ax = plot_train_test_errors(
        "$\ln\; \lambda$", lambdas, train_errors, test_errors)
    ax.set_xscale("log")
    plt.show()

def evaluate_number_of_data_points(degree, lambda_, num_points_sequence=None):
    """
    Evaluates and plots test & train error (RMSE) for different regularisation
    strength.
    Allows one to essentially recreate Figure 1.8 from Bishop.

    inputs
    ------
    degree - the degree of polynomial to fit
    lambda_ - the regularisation parameter if one is being used. Set to None
        if conventional least squares fit needed
    num_points_sequence - specifies how many points to use for each experiment
    """
    if num_points_sequence is None:
        num_points_sequence = [10,15,20,30,50,100,200]
    # sample test data once
    test_inputs, test_targets = sample_data(100, simple_sin)
    # 
    train_errors = []
    test_errors = []
    for N in num_points_sequence:
        # for each number of points fit the data then evaluate train/test error
        train_inputs, train_targets = sample_data(N, simple_sin)
        train_error, test_error = train_and_test(
            degree, train_inputs, train_targets, test_inputs,
            test_targets, lambda_)
        train_errors.append(train_error)
        test_errors.append(test_error)
    # plot the results
    fig, ax = plot_train_test_errors(
        "N", num_points_sequence, train_errors, test_errors)
    ax.set_xscale("log")
    plt.show()


def main(test_type, degree, lambda_, N):

    if test_type == 'degree':
        print("Evaluating the fit for different degrees of polynomial")
        evaluate_degree(lambda_, N)
    elif test_type == 'regulariser':
        evaluate_regulariser(degree, N)
    elif test_type == 'data':
        evaluate_number_of_data_points(degree, lambda_)
    else:
      raise ValueError("Unrecognised test-type")

def parse_command_line_args():
    # This function has been added to show how you might deal with command line
    # inputs in a more flexible way. I won't be expecting you to recreate
    # anything like this in the course. But if it is helpful to you, then please
    # use it.
    # this library helps us to process input commands
    import argparse
    parser = argparse.ArgumentParser(description='Polynomial Fitting: Evaluate polynomial degree or regulariser.')
    parser.add_argument(
        '--test-type', default='degree',
        help="Evaluate polynomial degree (degree), " 
            + "regulariser strength (regulariser) "
            + " or number of data-points (data).")
    parser.add_argument(
        '--degree', type=int, default=9, help="Default degree value")
    parser.add_argument(
        '--reg-param', type=float, default=None,
        help="Default regularisation strength")
    parser.add_argument(
        '--data', dest="N", type=int, default=10,
        help="Default number of data-points")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """
    This code can evaluate three influences on the quality of fit to your data:
      * polynomial degree 
      * regulariser strength 
      * number of data-points

    For degree run:
      python poly_fit_evaluate.py --test-type degree

    For regularisation strength run:
      python poly_fit_evaluate.py --test-type regulariser

    For number of data-points run:
      python poly_fit_evaluate.py --test-type data

    You can control the default: degree with the --degree flag, the
    regularisation strength with the --reg-param flag and the number of data
    points with the --data flag
    """
    args = parse_command_line_args()
    main(
        test_type=args.test_type, degree=args.degree, lambda_=args.lambda_,
        N=args.N)


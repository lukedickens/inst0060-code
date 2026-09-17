import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from fomlads.plot.distributions import overlay_2d_gaussian_contour

def plot_2d_data_and_approximating_gaussian(
        data, mu=None, Sigma=None, field_names=None):
    """
    Plots a collection of 2d datapoints as scatter plot, then fits maximum
    likelihood gaussian and overlays contours on the data.

    parameters
    ----------
    data -- Nx2 numpy.array of data points (each row is a data point, each
        column is a data dimension)
    field_names -- list/tuple of strings corresponding to data column labels
        If provided, the axes will be labelled with the field names.
    mu -- mean of approximating gaussian (1d array)
    Sigma -- covariance matrix of approximating gaussian (2d array) 
    """
    dim, N = data.shape
    # create a figure object and plot the data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ## TODO: You must complete this function so that it extracts the first and
    ## TODO: second column of the data array and scatter plots these.
    # if a mean (mu) vector and covariance matrix (Sigma) are supplied,
    # plot the contours of the associated gaussian
    if not (mu is None or Sigma is None):
        overlay_2d_gaussian_contour(ax, mu, Sigma)
    if not field_names is None:
        ## TODO: add the field names to each axis
        pass
    # make the spacing around the plot look nice
    plt.tight_layout()



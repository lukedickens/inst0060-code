"""
Provided code version for tutorial 04
"""
import numpy as np

def max_lik_1d_gaussian(samples):
    """
    parameters
    ----------
      samples - a 1d array of samples
    returns
    -------
      mu_ml - a maximum likelihood estimate of the mean
      sigma2_ml - a maximum likelihood estimate of the variance
    """
    # TODO: edit this function to calcualte the maximum likelihood parameters
    mu_ML = 0
    sigma2_ML = 0
    return mu_ML, sigma2_ML


def posterior_1d_gaussian(N, m_0, s2_0, mu_ml, sigma2):
    """
    parameters
    ----------
      N - number of samples
      m_0 - prior mean for data mean mu
      s2_0 - prior variance for data mean mu
      mu_ml - maximum likelihood mean
      sigma2 - known data variance

    returns
    -------
      m_N - posterior mean for data mean mu
      s2_N - posterior variance for data mean mu
    """
    # TODO: edit this function to calcualte the posterior parameters
    m_N = 0
    s2_N = 0
    return m_N, s2_N

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


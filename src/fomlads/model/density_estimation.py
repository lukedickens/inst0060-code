import numpy as np

def max_lik_1d_gaussian(samples):
    """
    parameters
    ----------
      samples - a vector of samples
    returns
    -------
      mu_ml - a maximum likelihood estimate of the mean
      sigma2_ml - a maximum likelihood estimate of the variance
    """
    N = samples.size
    mu_ml = np.sum(samples)/N
    sigma2_ml = np.sum((samples-mu_ml)**2)/N
    return mu_ml, sigma2_ml


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
    m_N = (sigma2*m_0 + N*s2_0*mu_ml)/(N*s2_0 + sigma2) 
    s2_N = 1/(1/s2_0 + N/sigma2)
    return m_N, s2_N


"""
Created 12th July 2022

@author : minh.ngo
"""

import numpy as np
from scipy.optimize import *

def gauss(x, A, mu, sigma):
    """
    Gaussian function

    Parameters
    -------------
    x : Any

    A : float
        Amplitude
    mu : float
        mean or centred point of LSF
    sigma : float
        sqrt of variance

    Returns
    ------------
    returned value : Any
                array_like or a number
    """
    return A * 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)

def fitted_gauss(wavelength, intensity):
    # Executing curve_fit on data
    parameters, covariance = curve_fit(gauss, wavelength, intensity)
    key = ['Amplitude', 'Mean', 'Sigma']
    return dict(zip(key, parameters))
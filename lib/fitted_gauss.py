"""
Created 12th July 2022

@author : minh.ngo
"""

import numpy as np
from scipy.optimize import leastsq

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
    return A / (sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)

def fitted_gauss(wavelength, intensity):
    # Executing curve_fit on data
    err_func = lambda params, x, y: gauss(x, *params) - y
    bins = np.linspace(min(wavelength), max(wavelength), len(wavelength))
    ind = np.argmin(abs(max(intensity) - intensity))
    mu = wavelength[ind]
    ind_half = np.argmin(abs(max(intensity)/2 - intensity))
    sigma = abs(wavelength[ind_half] - mu)
    init_intensity = 1 / (sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((bins - mu)/sigma)**2)
    A = max(intensity) / max(init_intensity)
    popt, ier = leastsq(err_func, x0=[A,mu,sigma], args=(wavelength, intensity))    
    key = ['Amplitude', 'Mean', 'Sigma']
    return dict(zip(key, popt))
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
    # Restore data of wavelength and intensity, after that we sort array by order of wavelength
    dtype=[('wavelength', np.float32), ('intensity', np.float32)]
    values = [(wavelength[i], intensity[i]) for i in range(len(intensity))]
    
    # Note : Use array_np.hstack()
    list_of_tuples = np.array(values, dtype=dtype)
    list_of_tuples = np.sort(list_of_tuples, order='wavelength')

    # Recast wavelength and intensity into numpy arrays so we can use their handy features
    wavelength_appro = np.asarray(list_of_tuples[:]['wavelength'])
    intensity_appro = np.asarray(list_of_tuples[:]['intensity'])

    # Executing curve_fit on data
    parameters, covariance = curve_fit(gauss, wavelength_appro, intensity_appro)
    intensity_appro = gauss(wavelength_appro, *parameters)
    key = ['Amplitude', 'Mean', 'Sigma']
    return dict(zip(key, parameters))
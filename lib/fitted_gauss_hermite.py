"""
Created 12th July 2022

@author : minh.ngo
"""

import numpy as np
from numpy.polynomial.hermite import hermfit, hermval

def fitted_gauss_hermite(wavelength, intensity, deg):
    # Restore data of wavelength and intensity, after that we sort array by order of wavelength
    dtype=[('wavelength', float), ('intensity', float)]
    values = [(wavelength[i], intensity[i]) for i in range(len(intensity))]
    
    # Note : Use array_np.hstack()
    list_of_tuples = np.array(values, dtype=dtype)
    list_of_tuples = np.sort(list_of_tuples, order='wavelength')

    # Recast wavelength and intensity into numpy arrays so we can use their handy features
    wavelength_appro = np.asarray(list_of_tuples[:]['wavelength'])
    intensity_appro = np.asarray(list_of_tuples[:]['intensity'])

    # Execute hermfit on data
    parameters = hermfit(wavelength_appro, intensity_appro, deg)
    intensity_appro = hermval(wavelength_appro, parameters)
    key = []
    for i in range(len(parameters)):
        key.append(f"Par{i}")
    return dict(zip(key, parameters))
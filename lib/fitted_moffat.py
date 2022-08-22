"""
Created 12th July 2022

@author : minh.ngo
"""

from lmfit.models import MoffatModel
#from scipy.optimize import leastsq
import numpy as np

def moffat(x, A, mu, sigma, beta):
    return A * (((x-mu)/sigma)**2 + 1)**(-beta)

def fitted_moffat(wavelength, intensity):    
    mod = MoffatModel()    
    init_pars = mod.guess(intensity, x=wavelength)
    out = mod.fit(intensity, init_pars, x=wavelength)
    valu = []
    for val in out.params.values():
        valu.append(val.value)
    dic = dict(zip(out.params.keys(), valu))
    dic.popitem()
    dic.popitem()
    return dic
    """
    mod = MoffatModel()    
    ind = np.argmin(abs(max(intensity) - intensity))
    A = max(intensity)
    mu = wavelength[ind]
    ind_half = np.argmin(abs(max(intensity)/2 - intensity))
    sigma = abs(wavelength[ind_half] - mu)
    pars = mod.make_params(amplitude=A, center=mu, sigma=sigma, beta=1)
    out = mod.fit(intensity, pars, x=wavelength)
    valu = []
    for val in out.params.values():
        valu.append(val.value)
    dic = dict(zip(out.params.keys(), valu))
    dic.popitem()
    dic.popitem()
    return dic
    """
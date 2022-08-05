"""
Created 12th July 2022

@author : minh.ngo
"""

from lmfit.models import MoffatModel
from scipy.optimize import leastsq
import numpy as np

def moffat(x, A, mu, sigma, beta):
    return A * (((x-mu)/sigma)**2 + 1)**(-beta)

def fitted_moffat(wavelength, intensity):
    err_func = lambda params, x, y: moffat(x, *params) - y
    beta = 1
    bins = np.linspace(min(wavelength), max(wavelength), len(wavelength))
    ind = np.argmin(abs(max(intensity) - intensity))
    mu = wavelength[ind]
    ind_half = np.argmin(abs(max(intensity)/2 - intensity))
    sigma = abs(wavelength[ind_half] - mu)
    init_intensity =  (((bins-mu)/sigma)**2 + 1)**(-beta)
    A = max(intensity) / max(init_intensity)
    popt, ier = leastsq(err_func, x0=[A,mu,sigma,beta], args=(wavelength, intensity)) 
    """
    mod = MoffatModel()
    pars = mod.guess(intensity, x=wavelength)
    out = mod.fit(intensity, pars, x=wavelength, method='least_squares')
    valu = []
    for val in out.params.values():
        valu.append(val.value)
    dic = dict(zip(out.params.keys(), valu))
    dic.popitem()
    dic.popitem()
    """
    li = ['amplitude', 'center', 'sigma', 'beta']
    return dict(zip(li, popt))
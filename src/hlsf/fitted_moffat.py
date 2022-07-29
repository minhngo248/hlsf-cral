"""
Created 12th July 2022

@author : minh.ngo
"""

from lmfit.models import MoffatModel

def moffat(x, A, mu, sigma, beta):
    return A * (((x-mu)/sigma)**2 + 1)**(-beta)

def fitted_moffat(wavelength, intensity):
    mod = MoffatModel()
    pars = mod.guess(intensity, x=wavelength)
    out = mod.fit(intensity, pars, x=wavelength, method='least_squares')
    valu = []
    for val in out.params.values():
        valu.append(val.value)
    dic = dict(zip(out.params.keys(), valu))
    dic.popitem()
    dic.popitem()
    return dic
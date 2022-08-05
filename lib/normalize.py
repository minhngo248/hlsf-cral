"""
Created 12th July 2022

@author : minh.ngo
"""

import numpy as np

def normalize(image_cut):
    """
    Normalize an image by uniform distribution

    Parameters
    ----------
    image_cut : array 2 dim

    Returns
    -----------
    image_uni : array 1 dim

    """  
    image_uni = np.ravel(image_cut)
    max = np.max(image_uni)
    min = np.min(image_uni)
    image_uni = (image_uni - min)/(max - min)
    return image_uni

def gaussian_normalize(waves, flux):
    max_flux = max(flux)
    ind = np.argmin(abs(max_flux-flux))
    mu = waves[ind]
    ind_half = np.argmin(abs(max_flux/2 - flux))
    half_wave = waves[ind_half]
    sigma = abs(half_wave - mu)
    normalized_flux =  1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((waves-mu)/sigma)**2)
    return normalized_flux
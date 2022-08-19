"""
Created 12th July 2022

@author : minh.ngo
"""

import numpy as np
from .fitted_gauss import *

def normalize(image_cut):
    r"""
    Normalize an image by uniform distribution
    ```{math}
    f(x) = \big \frac{x - min}{max - min}
    ```

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
    dic_params = fitted_gauss(waves, flux)
    A = dic_params['Amplitude']
    normalized_flux = flux / A
    return normalized_flux
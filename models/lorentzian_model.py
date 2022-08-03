"""
Created 12th July 2022

@author : minh.ngo
"""

from .lsf_model_2 import LSF_MODEL_2
from numpy.polynomial import polynomial as P
import numpy as np
from scipy.optimize import leastsq

def poly_lorentz(x, *args):
    if len(args) != 6:
        raise NameError("6 parameters needed")
    A = P.polyval(x, [args[0], args[1]])
    mu = P.polyval(x, [args[2], args[3]])
    sigma = P.polyval(x, [args[4], args[5]])
    return A/np.pi * (sigma / ((x-mu)**2 + sigma**2))

class LORENTZIAN_MODEL(LSF_MODEL_2):
    """
    Gaussian model
    """
    def __init__(self, lsf_data, listLines=None, _popt=None) -> None:
        super().__init__(lsf_data, listLines, _popt)
        list_waves = []
        list_intensity = []
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                data = self.lsf_data[i].get_data_line(nb_line)
                map_wave = data['map_wave']
                waveline = data['waveline']
                intensity = data['intensity']
                list_waves.append(map_wave-waveline)
                list_intensity.append(intensity)
        list_waves = [i for item in list_waves for i in item]
        list_waves = np.array(list_waves)
        list_intensity = [i for item in list_intensity for i in item]
        list_intensity = np.array(list_intensity)     
        # wavelines : save all wavelength of lines
        self._wavelines = []
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                waveline = self.lsf_data[i].get_data_line(nb_line)['waveline']
                self._wavelines.append(waveline)
        self._wavelines = np.array(self._wavelines)
        if (_popt == None):
            # Calculate linear coeff for 3 parameters
            err_func = lambda params, x, y: poly_lorentz(x, *params) - y
            popt, ier = leastsq(err_func, x0=[2,4,1,5,4,4], args=(list_waves, list_intensity))
            self._popt = popt

    def evaluate_intensity(self, w_0, waves):
        """
        Evaluate data from parameters 

        Parameters
        -------------
        w_0             : float
                        point whose intensity is max
        waves           : array-like

        Returns
        -----------
        eval_intensity  : array-like
        """
        eval_intensity = poly_lorentz(waves-w_0, *self._popt)
        return eval_intensity

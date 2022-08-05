"""
Created 12th July 2022

@author : minh.ngo
"""

from .lsf_model import LSF_MODEL
from numpy.polynomial import polynomial as P
import numpy as np
from scipy.optimize import leastsq

def poly_gauss(x, w_0, *args):
    if len(args) != 6:
        raise NameError("6 parameters needed")
    A = P.polyval(w_0, [args[0], args[1]])
    mu = P.polyval(w_0, [args[2], args[3]])
    sigma = P.polyval(w_0, [args[4], args[5]])
    return A / (sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)

class GAUSSIAN_MODEL_2(LSF_MODEL):
    """
    Gaussian model
    """
    def __init__(self, lsf_data, listLines=None, _coeff=None) -> None:
        super().__init__(lsf_data, listLines, _coeff)    
        # 6 params
        array_waves = np.empty(0, dtype=float)
        array_w_0 = np.empty(0, dtype=float)
        array_intensity = np.empty(0, dtype=float)
        for k in range(len(self.lsf_data)):
            for nb_line in self._listLines[k]:
                data = self.lsf_data[k].get_data_line(nb_line)
                map_wave = data['map_wave']
                waveline = data['waveline']
                intensity = data['intensity']
                array_waves = np.concatenate((array_waves, map_wave))
                w_0 = np.full_like(map_wave, waveline)
                array_w_0 = np.concatenate((array_w_0, w_0))
                array_intensity = np.concatenate((array_intensity, intensity))
        self.array_waves = array_waves
        self.array_w_0 = array_w_0
        self.array_intensity = array_intensity

        # wavelines : save all wavelength of lines
        self._wavelines = []
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                waveline = self.lsf_data[i].get_data_line(nb_line)['waveline']
                self._wavelines.append(waveline)
        self._wavelines = np.array(self._wavelines)
        
        if (_coeff == None):
            err_func = lambda params, x, w_0, y: poly_gauss(x, w_0, *params) - y
            bins = np.linspace(min(array_waves-array_w_0), max(array_waves-array_w_0), len(array_waves))
            ind = np.argmin(abs(max(array_intensity)-array_intensity))
            mu = array_waves[ind]-array_w_0[ind]
            ind_half = np.argmin(abs(max(array_intensity)/2 - array_intensity))
            sigma = abs(array_waves[ind_half] - array_w_0[ind_half] - mu)
            init_intensity = 1 / (sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((bins - mu)/sigma)**2)
            A = max(array_intensity) / max(init_intensity)
            popt, ier = leastsq(err_func, x0=[A,0,mu,0,sigma,0], args=(array_waves-array_w_0, array_w_0, array_intensity))
            self._coeff = popt

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
        eval_intensity = poly_gauss(waves-w_0, w_0, *self._coeff)
        return eval_intensity

"""
Created 12th July 2022

@author : minh.ngo
"""

from .lsf_model import LSF_MODEL
from numpy.polynomial import polynomial as P
import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.optimize import leastsq

def poly_gauss_hermite(x, w_0, deg, *args):
    if len(args) != (deg+1)*2:
        raise NameError("Wrong Parameters")
    params = [P.polyval(w_0, [args[i], args[i+1]]) for i in range(0, 2*deg+1, 2)]
    return hermval(x, params, tensor=False)

class GAUSS_HERMITE_MODEL_2(LSF_MODEL):
    """
    Gauss Hermite model
    """
    def __init__(self, lsf_data, deg, listLines=None, _coeff=None) -> None:
        super().__init__(lsf_data, listLines, _coeff)
        self.deg = deg
        array_waves = np.empty(0, dtype=float)
        array_w_0 = np.empty(0, dtype=float)
        array_intensity = np.empty(0, dtype=float)
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                data = self.lsf_data[i].get_data_line(nb_line)
                map_wave = data['map_wave']
                waveline = data['waveline']
                intensity = data['intensity']
                array_waves = np.concatenate((array_waves, map_wave))
                w_0 = np.full_like(map_wave, waveline)
                array_w_0 = np.concatenate((array_w_0, w_0))
                array_intensity = np.concatenate((array_intensity, intensity))
    
        # wavelines : save all wavelength of lines
        self._wavelines = []
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                waveline = self.lsf_data[i].get_data_line(nb_line)['waveline']
                self._wavelines.append(waveline)
        self._wavelines = np.array(self._wavelines)
        if (_coeff == None):
            # Calculate linear coeff for 3 parameters
            err_func = lambda params, x, w_0, y: poly_gauss_hermite(x, w_0, deg, *params) - y
            p = np.ones(2*(deg+1))
            popt, ier = leastsq(err_func, x0=p, args=(array_waves-array_w_0, array_w_0, array_intensity))
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
        eval_intensity = poly_gauss_hermite(waves-w_0, w_0, self.deg, *self._coeff)
        return eval_intensity

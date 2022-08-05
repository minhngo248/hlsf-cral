"""
Created 12th July 2022

@author : minh.ngo
"""

from .lsf_model import LSF_MODEL
from numpy.polynomial import polynomial as P
import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.optimize import leastsq

def poly_gauss_hermite(x, deg, *args):
    if len(args) != (deg+1)*2:
        raise NameError("Wrong Parameters")
    params = [P.polyval(x, [args[i], args[i+1]]) for i in range(0, 2*deg+1, 2)]
    return hermval(x, params, tensor=False)

class GAUSS_HERMITE_MODEL_2(LSF_MODEL):
    """
    Gauss Hermite model
    """
    def __init__(self, lsf_data, deg, listLines=None, _coeff=None) -> None:
        super().__init__(lsf_data, listLines, _coeff)
        self.deg = deg
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
        if (_coeff == None):
            # Calculate linear coeff for 3 parameters
            err_func = lambda params, x, y: poly_gauss_hermite(x, deg, *params) - y
            p = np.ones(2*(deg+1))
            popt, ier = leastsq(err_func, x0=p, args=(list_waves, list_intensity))
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
        eval_intensity = poly_gauss_hermite(waves-w_0, self.deg, *self._coeff)
        return eval_intensity

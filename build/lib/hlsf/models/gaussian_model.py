"""
Created 12th July 2022

@author : minh.ngo
"""

from numpy.polynomial import polynomial as P
import numpy as np
from .lsf_model import LSF_MODEL
from ..lib.fitted_gauss import fitted_gauss

class GAUSSIAN_MODEL(LSF_MODEL):
    """
    Gaussian model
    """
    def __init__(self, lsf_data, listLines=None, _params_linear=None) -> None:
        super().__init__(lsf_data, listLines, _params_linear)
        # dic_params : save all parameters of each line
        self._dic_params = {'Amplitude': [], 'Mean': [], 'Sigma': []}
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                data = self.lsf_data[i].get_data_line(nb_line)
                map_wave = data['map_wave']
                waveline = data['waveline']
                intensity = data['intensity']
                params = fitted_gauss(map_wave - waveline, intensity)
                for key in params.keys():
                    self._dic_params[key].append(params[key])      
        # wavelines : save all wavelength of lines
        self._wavelines = []
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                waveline = self.lsf_data[i].get_data_line(nb_line)['waveline']
                self._wavelines.append(waveline)
        self._wavelines = np.array(self._wavelines)
        if (_params_linear == None):
            # Calculate linear coeff for 3 parameters
            params_linear = {'Amplitude': None, 'Mean': None, 'Sigma': None}
            for key in self._dic_params.keys():
                if len(self._wavelines) > 1:
                    coeff = P.polyfit(self._wavelines, self._dic_params[key], deg=1)
                elif len(self._wavelines) == 1:
                    coeff = P.polyfit(self._wavelines, self._dic_params[key], deg=0)
                params_linear[key] = coeff
            self._params_linear = params_linear

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
        A = P.polyval(waves, self._params_linear['Amplitude'])
        mu = P.polyval(waves, self._params_linear['Mean'])
        sigma = P.polyval(waves, self._params_linear['Sigma'])
        eval_intensity = A * 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((waves-w_0-mu)/sigma)**2)
        return eval_intensity

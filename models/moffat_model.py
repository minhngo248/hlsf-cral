"""
Created 12th July 2022

@author : minh.ngo
"""

from ..lib.fitted_moffat import fitted_moffat
from .lsf_model import LSF_MODEL
import numpy as np
from numpy.polynomial import polynomial as P

class MOFFAT_MODEL(LSF_MODEL):
    def __init__(self, lsf_data, listLines=None, _params_linear=None) -> None:
        super().__init__(lsf_data, listLines, _params_linear)
        self._dic_params = {'amplitude': [], 'center': [], 'sigma': [], 'beta': []}
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                data = self.lsf_data[i].get_data_line(nb_line)
                map_wave = data['map_wave']
                waveline = data['waveline']
                intensity = data['intensity']
                params = fitted_moffat(map_wave - waveline, intensity)
                for key in params.keys():
                    self._dic_params[key].append(params[key])        
        self._wavelines = []
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                waveline = self.lsf_data[i].get_data_line(nb_line)['waveline']
                self._wavelines.append(waveline)
        self._wavelines = np.array(self._wavelines)
        if (_params_linear == None):
            params_linear = {'amplitude': None, 'center': None, 'sigma': None, 'beta': None}
            for key in self._dic_params.keys():
                if len(self._wavelines) > 1:
                    coeff, stats = P.polyfit(self._wavelines, self._dic_params[key], deg=1, full=True)
                elif len(self._wavelines) == 1:
                    coeff, stats = P.polyfit(self._wavelines, self._dic_params[key], deg=0, full=True)
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
        A = P.polyval(waves, self._params_linear['amplitude'])
        mu = P.polyval(waves, self._params_linear['center'])
        sigma = P.polyval(waves, self._params_linear['sigma'])
        beta = P.polyval(waves, self._params_linear['beta'])
        eval_intensity = A * (((waves-w_0-mu)/sigma)**2 + 1)**(-beta)
        return eval_intensity
    
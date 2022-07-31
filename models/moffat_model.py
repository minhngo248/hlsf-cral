"""
Created 12th July 2022

@author : minh.ngo
"""

from ..lib.fitted_moffat import fitted_moffat
from .lsf_model import LSF_MODEL
import numpy as np
from lmfit.models import LinearModel

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
        mod = LinearModel()
        self._wavelines = []
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                waveline = self.lsf_data[i].get_data_line(nb_line)['waveline']
                self._wavelines.append(waveline)
        self._wavelines = np.array(self._wavelines)
        if (_params_linear == None):
            if len(self._wavelines) > 1:
                params_linear = {'amplitude': None, 'center': None, 'sigma': None, 'beta': None}
                for key, value in self._dic_params.items():
                    pars = mod.guess(value, x=self._wavelines)
                    out = mod.fit(value, pars, x=self._wavelines)
                    params_linear[key] = [out.params['slope'].value, out.params['intercept'].value]
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
        if len(self._wavelines) == 1:
            A = self._dic_params["amplitude"][0]
            mu = self._dic_params["center"][0]
            sigma = self._dic_params["sigma"][0]
            beta = self._dic_params["beta"][0]
        elif len(self._wavelines) > 1:
            A = self._params_linear["amplitude"][0] * waves + self._params_linear["amplitude"][1]
            mu = self._params_linear["center"][0] * waves + self._params_linear["center"][1]
            sigma = self._params_linear["sigma"][0] * waves + self._params_linear["sigma"][1]
            beta = self._params_linear["beta"][0] * waves + self._params_linear["beta"][1]
        eval_intensity = A * (((waves-w_0-mu)/sigma)**2 + 1)**(-beta)
        return eval_intensity
    
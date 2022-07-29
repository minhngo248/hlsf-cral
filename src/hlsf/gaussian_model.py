"""
Created 12th July 2022

@author : minh.ngo
"""

from lmfit.models import LinearModel
import numpy as np
from .lsf_model import LSF_MODEL
from .fitted_gauss import fitted_gauss

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
        mod = LinearModel()
        # wavelines : save all wavelength of lines
        self._wavelines = []
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                waveline = self.lsf_data[i].get_data_line(nb_line)['waveline']
                self._wavelines.append(waveline)
        self._wavelines = np.array(self._wavelines)
        if (_params_linear == None):
            # Calculate linear coeff for 3 parameters
            if len(self._wavelines) > 1:
                params_linear = {'Amplitude': None, 'Mean': None, 'Sigma': None}
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
            A = self._dic_params["Amplitude"][0]
            mu = self._dic_params["Mean"][0]
            sigma = self._dic_params["Sigma"][0]
        elif len(self._wavelines) > 1:
            A = self._params_linear["Amplitude"][0] * waves + self._params_linear["Amplitude"][1]
            mu = self._params_linear["Mean"][0] * waves + self._params_linear["Mean"][1]
            sigma = self._params_linear["Sigma"][0] * waves + self._params_linear["Sigma"][1]
        eval_intensity = A * 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((waves-w_0-mu)/sigma)**2)
        return eval_intensity

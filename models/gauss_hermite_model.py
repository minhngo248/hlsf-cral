"""
Created 12th July 2022

@author : minh.ngo
"""

from ..lib.fitted_gauss_hermite import fitted_gauss_hermite
from .lsf_model import LSF_MODEL
import numpy as np
from numpy.polynomial import polynomial as P
from numpy.polynomial.hermite import hermval

def intensity(x, params_linear: dict):
    params = [P.polyval(x, val) for val in params_linear.values()]
    y = hermval(x, params)
    return y

class GAUSS_HERMITE_MODEL(LSF_MODEL):
    def __init__(self, lsf_data, deg, listLines=None, _coeff=None) -> None:
        """
        Constructor

        Parameters:
        --------------
        lsf_data        : array-like(LSF_DATA) or LSF_DATA
                        data needed save in this object
        deg             : int
                        degree of Hermite polynominal
        listLines       : a number or ordered array-like
                        sequence of indice of lines (0 - 254), list or nested list
                        ex : 10, [5, 6, 9], [[5,6,9], [1,2]]
        _params_linear  : dict[str, list[float]]
                        dictionary used for saving 2 coeff of a line after evaluating
                        parameters by LinearModel
                        Ex : {"Par0": [
                                    -9.365769219685553e-06,
                                    2.527443909813685
                                    ],
                                "Par1": [
                                    -3.058392718578551e-07,
                                    0.005943723524430775
                                    ],
                                "Par2": [
                                    -4.5539365345020435e-06,
                                    1.0077788675153003
                                ]}
        """ 
        super().__init__(lsf_data, listLines, _coeff)
        self.deg = deg            
        list_params = [f'Par{i}' for i in range(deg+1)]    
        self._dic_params = dict.fromkeys(list_params, [])

        # save all parameters in 2 dimensional array
        concat_params = np.empty((len(self._dic_params), 0), dtype=np.float64)
        for k in range(len(self._listLines)):
            array_params = np.zeros((len(self._dic_params), len(self._listLines[k])))
            for i, nb_line in enumerate(self._listLines[k]):
                data = self.lsf_data[k].get_data_line(nb_line)
                map_wave = data['map_wave']
                waveline = data['waveline']
                intensity = data['intensity']  
                params = fitted_gauss_hermite(map_wave - waveline, intensity, deg)
                for j, key in enumerate(params.keys()):
                    array_params[j, i] = params[key]
            concat_params = np.concatenate((concat_params, array_params), axis=1)

        self._dic_params = dict(zip(list_params, concat_params))
        self._wavelines = []
        for i in range(len(self.lsf_data)):
            for nb_line in self._listLines[i]:
                waveline = self.lsf_data[i].get_data_line(nb_line)['waveline']
                self._wavelines.append(waveline)
        self._wavelines = np.array(self._wavelines)
        if (_coeff == None):          
            params_linear = dict.fromkeys(list_params, None)
            for key in self._dic_params.keys():
                if len(self._wavelines) > 1:
                    coeff = P.polyfit(self._wavelines, self._dic_params[key], deg=1)
                elif len(self._wavelines) == 1:
                    coeff = P.polyfit(self._wavelines, self._dic_params[key], deg=0)
                params_linear[key] = coeff
            self._coeff = params_linear  

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
        params = [P.polyval(waves, self._coeff[key]) for key in self._coeff.keys()]
        #eval_intensity = [intensity(x-w_0, self._params_linear) for x in waves]
        eval_intensity = hermval(waves-w_0, params, tensor=False)
        return eval_intensity
    
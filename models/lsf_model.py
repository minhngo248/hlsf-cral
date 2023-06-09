"""
Created 12th July 2022

@author : minh.ngo
"""

from .lsf_data import LSF_DATA
import numpy as np
from numpyencoder import NumpyEncoder
from numpy.polynomial import polynomial as P
import json
import importlib
import matplotlib.pyplot as plt
from ..lib.error import *

def str_to_class(modu, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(modu)
    # get the class, will raise KeyError if class cannot be found
    c = m.__dict__[class_name]
    return c

class LSF_MODEL(object):
    """
    Model for evaluating LSF function
    """
    def __init__(self, lsf_data, listLines, _coeff: dict) -> None:
        """
        Constructor

        Parameters:
        --------------
        lsf_data        : array-like(LSF_DATA) or LSF_DATA
                        data needed save in this object
        listLines       : a number or ordered array-like
                        sequence of indice of lines, list or nested list
                        ex : for one lamp: 10, [5, 6, 9]
                        for numerous lamps :[[5,6,9], [1,2], ...]
        _coeff          : dict[str, list[float]]
                        dictionary used for saving 2 coeff of a line after evaluating
                        parameters by LinearModel
                        Ex : Gaussian model {"Amplitude": [
                                                -9.365769219685553e-06,
                                                2.527443909813685
                                            ],
                                            "Mean": [
                                                -3.058392718578551e-07,
                                                0.005943723524430775
                                            ],
                                            "Sigma": [
                                                -4.5539365345020435e-06,
                                                1.0077788675153003
                                            ]}
        """        
        self._coeff = _coeff     
        if listLines == None:
            self.lsf_data = np.asarray([lsf_data])  if type(lsf_data) == LSF_DATA else lsf_data
            li = np.empty(len(self.lsf_data), dtype=np.ndarray)
            for i in range(len(li)):
                li[i] = np.array(list(self.lsf_data[i].get_line_list().keys()))
            self._listLines = li
        else:
            if type(lsf_data) == LSF_DATA:
                self.lsf_data = np.asarray([lsf_data])
                self._listLines = np.asarray([[listLines]]) if type(listLines) == int else np.asarray([listLines]) 
            else:
                self.lsf_data = lsf_data
                self._listLines = listLines  
        for i in range(1, len(self.lsf_data)):
            if self.lsf_data[i] != self.lsf_data[0]:
                raise NameError("Config, detID or type de normalization did not work")

    @classmethod
    def from_json(obj, filename):
        """
        Constructor from a JSON file

        Parameters
        ------------
        filename    : str
                    ex : gaussian_model_H_Xe.json
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        dic_lsf_data = data['lsf_data']
        lsf_data = [LSF_DATA.from_dict(lsf) for lsf in dic_lsf_data]
        obj = str_to_class('hlsf.models', data['name'])
        _coeff = data['coeff']
        try:
            line = data['line']
        except KeyError:
            if 'GAUSS_HERMITE_MODEL' in data['name']:
                return obj(lsf_data, deg=len(_coeff)-1, _coeff=_coeff)
            return obj(lsf_data, _coeff=_coeff)
        else:
            if 'GAUSS_HERMITE_MODEL' in data['name']:
                return obj(lsf_data, deg=len(_coeff)-1, _coeff=_coeff)
            return obj(lsf_data, [[line]], _coeff=_coeff)

    def plot(self, w_0, waves, ax, centre=True):
        """
        Parameters
        -----------
        w_0     : float
                wavelength of line
        waves   : array-like
                wavelength of each point
        ax      : matplotlib.pyplot.axes
                ax for plotting figures
        centre  : bool
                Real wavelength : False, relative wavelength centered in 0 : True
        """
        max_wave = max(waves-w_0)
        min_wave = min(waves-w_0)
        wave_linspace = np.linspace(min_wave, max_wave, len(waves))
        eval_intensity = self.evaluate_intensity(w_0, wave_linspace+w_0)
        ax.set_xlabel(r'wavelength ($\AA$)')
        ax.set_ylabel('intensity')
        if centre:
            ax.plot(wave_linspace, eval_intensity, label=f"{self.__class__.__name__} fit")
        else:
            ax.plot(wave_linspace+w_0, eval_intensity, label=f"{self.__class__.__name__} fit")
        ax.grid()
        ax.legend()
        ax.set_title(f'{str.lower(self.__class__.__name__).replace("_"," ").capitalize()}')

    def plot_delta(self, w_0, delta_w, ax, centre=True):
        """
        Parameters
        -----------
        w_0     : float
                wavelength of line
        delta_w : positive float
                largeur of wavelength wanted to visualize
        ax      : matplotlib.pyplot.axes
        centre  : bool
                Real wavelength : False, relative wavelength centered in 0 : True
        """
        wave_linspace = np.linspace(-delta_w, delta_w, int(300*delta_w))
        eval_intensity = self.evaluate_intensity(w_0, wave_linspace+w_0)
        ax.set_xlabel(r'wavelength ($\AA$)')
        ax.set_ylabel('intensity')
        if centre:
            ax.plot(wave_linspace, eval_intensity, label=f"{self.__class__.__name__} fit")
        else:
            ax.plot(wave_linspace+w_0, eval_intensity, label=f"{self.__class__.__name__} fit")
        ax.grid()
        ax.legend()
        ax.set_title(f'{str.lower(self.__class__.__name__).replace("_"," ").capitalize()}')

    def error_rms(self, lsf_data: LSF_DATA, listLines):
        """
        Calculate RMS error of all lines in LSF_DATA

        Parameters
        ------------
        lsf_data    : LSF_DATA
                    data of a slice for evaluating intensity
        listLines   : int or array-like[int]
                    list of lines in a slice extracted from FITS or TXT file
                    ex : 9, [9, 10, 56]

        Returns
        ---------
        err         : float or list[float]
                    RMS error of all lines
        """
        listLines = [listLines] if type(listLines) == int else np.asarray(listLines)
        err = []
        for nb_line in listLines:
            w_0 = lsf_data._listLines[nb_line]
            data = lsf_data.get_data_line(nb_line)
            waves = data['map_wave']
            intensity = data['intensity']
            eval_intensity = self.evaluate_intensity(w_0, waves)
            err.append(rms_error(intensity, eval_intensity))
        if len(err) == 1:
            err = err[0]
        return err

    def plot_error_rms(self, lsf_data: LSF_DATA, ax, listLines=None):
        """
        Parameters
        ------------
        lsf_data    : LSF_DATA
                    data of a slice for evaluating intensity
        ax          : matplotlib.pyplot.axes
        listLines   : int or array-like[int]
                    list of lines in a slice extracted from FITS or TXT file
                    ex : 9, [9, 10, 56]
        """
        if isinstance(listLines, (np.ndarray, range, list)):            
            listLines = [listLines] if type(listLines) == int else np.asarray(listLines)
        else:
            listLines = np.arange(lsf_data._lineUp, lsf_data._lineDown+1)
        err = self.error_rms(lsf_data, listLines)
        wavelength_line = [lsf_data.get_data_line(nb_line)['waveline'] for nb_line in listLines]
        ax.plot(wavelength_line, err)
        

    def error_max(self, lsf_data: LSF_DATA, listLines):
        """
        Calculate Max relative error of all lines in LSF_DATA

        Parameters
        ------------
        lsf_data    : LSF_DATA
                    data of a slice for evaluating intensity
        listLines   : int or array-like[int]
                    list of lines in a slice extracted from FITS or TXT file
                    ex : 9, [9, 10, 56]

        Returns
        ---------
        err         : float or list[float]
                    Max relative error of all lines
        """
        listLines = [listLines] if type(listLines) == int else np.asarray(listLines)
        err = []
        for nb_line in listLines:
            w_0 = lsf_data._listLines[nb_line]
            data = lsf_data.get_data_line(nb_line)
            waves = data['map_wave']
            intensity = data['intensity']
            eval_intensity = self.evaluate_intensity(w_0, waves)
            err.append(max_relative_error(intensity, eval_intensity))
        if len(err) == 1:
            err = err[0]
        return err

    def plot_error_max(self, lsf_data: LSF_DATA, ax, listLines=None):
        """
        Parameters
        ------------
        lsf_data    : LSF_DATA
                    data of a slice for evaluating intensity
        ax          : matplotlib.pyplot.axes        
        listLines   : int or array-like[int]
                    list of lines in a slice extracted from FITS or TXT file
                    ex : 9, [9, 10, 56]
        """
        ax.set_xlabel(r'wavelength ($\AA$)')
        ax.set_ylabel('Max relative err')
        if isinstance(listLines, (np.ndarray, range, list)):            
            listLines = [listLines] if type(listLines) == int else np.asarray(listLines) 
        else:
            listLines = np.arange(lsf_data._lineUp, lsf_data._lineDown+1)
        err = self.error_max(lsf_data, listLines)
        wavelength_line = [lsf_data.get_data_line(nb_line)['waveline'] for nb_line in listLines]
        ax.plot(wavelength_line, err)

    def plot_parameters(self):
        """
        Plot all parameters of each models
        and a fitted-line
        """
        if len(self._coeff) <= 4:
            fig, axes = plt.subplots(len(self._coeff), 1)
            plt.xlabel(r"wavelength ($\AA$)")
            if "_2" in self.__class__.__name__:
                # Model 2
                for i, key in enumerate(self._coeff.keys()):
                    axes[i].plot(self._wavelines, P.polyval(self._wavelines, self._coeff[key]), '+', color='green', label='Model 2')
            else:
                for i, key in enumerate(self._coeff.keys()):
                    axes[i].scatter(self._wavelines, self._dic_params[key], marker='o')
                    axes[i].plot(self._wavelines, P.polyval(self._wavelines, self._coeff[key]), color='red')
                    axes[i].set_ylabel(key)
                    axes[i].grid()
            fig.suptitle(f"{self.__class__.__name__.replace('_',' ').capitalize()}") 
            plt.legend()
            plt.show() 
        else:
            shape = (3, 4)
            fig, axes = plt.subplots(3, 4)
            plt.xlabel(r"wavelength ($\AA$)")
            if "_2" in self.__class__.__name__:
                # Model 2
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        axes[i, j].plot(self._wavelines, P.polyval(self._wavelines, self._coeff[f'Par{shape[1]*i+j}']), '+', color='green', label='Model 2')
            else:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        axes[i, j].scatter(self._wavelines, self._dic_params[f'Par{shape[1]*i+j}'], marker='o')
                        axes[i, j].plot(self._wavelines, P.polyval(self._wavelines, self._coeff[f'Par{shape[1]*i+j}']), color='red')
                        axes[i, j].set_ylabel(f'Par{shape[1]*i+j}')
                        axes[i, j].grid()
            fig.suptitle(f"{self.__class__.__name__.replace('_',' ').capitalize()}") 
            plt.legend()
            plt.show() 

    def write_json(self, filename):
        """
        Serialize into a file JSON

        Parameters
        -----------
        filename    : str
                    ex : gaussian_model_H_Ar.json
        """ 
        classname = self.__class__.__name__
        dic_lsf_data = [lsf.to_dict() for lsf in self.lsf_data]
        if len(self._wavelines) > 1:
            dic = {'name': classname, 'coeff' : self._coeff, 'lsf_data': dic_lsf_data}
        elif len(self._wavelines) == 1:
            dic = {'name': classname, 'coeff' : self._coeff, 'line': self._listLines[0][0], 'lsf_data': dic_lsf_data}
        json_object = json.dumps(dic, indent=4, cls=NumpyEncoder)
        with open(filename, "w") as outfile:
            outfile.write(json_object)        

    def __del__(self):
        """
        Destructor
        """
        print("Destructor called")
        
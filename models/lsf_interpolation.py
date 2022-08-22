import numpy as np
from scipy import interpolate
from scipy import ndimage
import matplotlib.pyplot as plt
from ..lib.error import *
from . import LSF_DATA

class LSF_INTERPOLATION(object):
    def __init__(self, list_lsf_data, listLines=None) -> None:
        """
        Constructor

        Parameters:
        --------------
        lsf_data                : LSF_DATA for evaluation
                                linspace256
        list_lsf_data           : array-like(LSF_DATA) or LSF_DATA
                                data needed save in this object
        listLines               : a number or ordered array-like
                                sequence of indice of lines, list or nested list
                                ex : for one lamp: 10, [5, 6, 9]
                                for numerous lamps :[[5,6,9], [1,2], ...]
        """            
        if listLines == None:
            if type(list_lsf_data) == LSF_DATA:
                self.list_lsf_data = np.asarray([list_lsf_data])  
            else:
                self.list_lsf_data = list_lsf_data
            li = np.empty(len(self.list_lsf_data), dtype=np.ndarray)
            for i in range(len(li)):
                li[i] = np.array(list(self.list_lsf_data[i].get_line_list().keys()))
            self._listLines = li
        else:
            if type(list_lsf_data) == LSF_DATA:
                self.list_lsf_data = np.asarray([list_lsf_data])
                if type(listLines) == int:
                    self._listLines = np.asarray([[listLines]]) 
                else: 
                    self._listLines = np.asarray([listLines]) 
            else:
                self.list_lsf_data = list_lsf_data
                self._listLines = listLines  
        for i in range(1, len(self.list_lsf_data)):
            if self.list_lsf_data[i] != self.list_lsf_data[0]:
                raise NameError("Config, detID or type de normalization did not work")


    def get_all_data(self):
        """
        Get relative wavelength, wavelength of all lines, intensity in
        the slice

        Returns
        ------------
        dic         : dict['array_pos': relative wavelength,
                            'array_waves': wavelength of all lines,
                            'array_intensity': intensity]
        """
        array_waves = np.empty(0, dtype=float)
        array_intensity = np.empty(0, dtype=float)
        array_pos = np.empty(0, dtype=float)
        for lsf in self.list_lsf_data:
            data = lsf.get_all_data()
            array_pos = np.concatenate((array_pos, data['array_pos']))
            array_waves = np.concatenate((array_waves, data['array_waves']))
            array_intensity = np.concatenate((array_intensity, data['array_intensity']))
        return {'array_pos': array_pos, 'array_waves': array_waves, 'array_intensity': array_intensity}

    def interpolate_data(self, method='linear', step_pos=1e-2, step_wave=10):
        """ 
        Parameters
        -----------
        step_pos    : float
                    distance of relative wavelength ($\overset{\circ}{A}$) for x-coor of image
        step_wave   : float
                    delta of wavelength ($\overset{\circ}{A}$) for y-coor of image

        Returns
        ------------
        dic         : dict
                    x : x-axis of image
                    y : y-axis of image
                    grid_z : image after being interpolated
        """
        if len(self._listLines) <= 1:
            raise NameError(f"Cannot interpolate, not enough lines")
        data = self.get_all_data()
        array_pos = data['array_pos']
        array_waves = data['array_waves']
        array_intensity = data['array_intensity']
        x = np.arange(min(array_pos), max(array_pos), step_pos)
        y = np.arange(min(array_waves), max(array_waves), step_wave)
        grid_x, grid_y = np.meshgrid(x, y)
        grid_z = interpolate.griddata(np.array([array_pos, array_waves]).T, array_intensity, (grid_x, grid_y), method=method)
        return {'x': x, 'y': y, 'grid_z': grid_z}

    def plot_interpolate_data(self, method='linear', step_pos=1e-2, step_wave=10):
        """
        Visualize image after interpolating

        Parameters
        -------------
        method      : str
                    'linear', 'cubic', 'nearest'
        step_pos    : float
                    delta of relative wavelength ($\overset{\circ}{A}$) for x-coor of image
        step_wave   : float
                    delta of wavelength ($\overset{\circ}{A}$) for y-coor of image
        """
        data = self.interpolate_data(method, step_pos, step_wave)
        x = data['x']
        y = data['y']
        grid_z = data['grid_z']
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xlabel(r'pos ($\AA$)')
        ax.set_ylabel(r'wavelength of line ($\AA$)')
        c = ax.pcolormesh(x, y, grid_z[:-1, :-1])
        plt.colorbar(c, ax=ax, label='interpolated intensity')
        plt.show()

    def evaluate_intensity(self, w_0, waves):
        """
        Evaluation of intensity from a number of line in the slice
        """
        interpolated_data = self.interpolate_data()
        x_coor_left = np.argmin(abs(interpolated_data['x']-min(waves-w_0)))
        x_coor_right = np.argmin(abs(interpolated_data['x']-max(waves-w_0)))
        y_coor = np.argmin(abs(interpolated_data['y']-w_0))
        x_coor = np.linspace(x_coor_left, x_coor_right, len(waves))
        y_coor = np.full_like(x_coor, y_coor)
        evaluated_intensity = ndimage.map_coordinates(interpolated_data['grid_z'], [y_coor, x_coor], order=1)
        pos = np.linspace(interpolated_data['x'][x_coor_left], interpolated_data['x'][x_coor_right], len(evaluated_intensity))
        return pos, evaluated_intensity

    def plot_evaluated_intensity(self, lsf_data: LSF_DATA, nb_line, ax, centre=True):
        data = lsf_data.get_data_line(nb_line)
        waves = data['map_wave']
        w_0 = data['waveline']
        pos, evaluated_intensity = self.evaluate_intensity(w_0, waves)
        if centre:
            ax.plot(pos, evaluated_intensity, label='Bspline data')
        else: 
            ax.plot(pos+w_0, evaluated_intensity, label='Bspline data')

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
        if type(listLines) == int:
            listLines = [listLines]
        else:
            listLines = np.asarray(listLines)
        err = []
        for nb_line in listLines:
            w_0 = lsf_data._listLines[nb_line]
            data = lsf_data.get_data_line(nb_line)
            waves = data['map_wave']
            intensity = data['intensity']
            # Sort
            tup = [(waves[i], intensity[i]) for i in range(len(waves))]
            array_tup = np.array(tup, dtype=[('waves', float), ('intensity', float)])
            array_tup = np.sort(array_tup, order='waves')
            intensity = array_tup['intensity']
            # Take evaluated intensity
            eval_intensity = self.evaluate_intensity(w_0, waves)[1]
            mask = ~np.isnan(eval_intensity)
            eval_intensity = eval_intensity[mask]
            intensity = intensity[mask]
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
            if type(listLines) == int:
                listLines = [listLines]
            else:
                listLines = np.asarray(listLines)
        else:
            listLines = np.arange(self._lineUp, self._lineDown+1)
        err = self.error_rms(listLines)
        wavelength_line = [self.get_data_line(nb_line)['waveline'] for nb_line in listLines]
        ax.plot(wavelength_line, err)
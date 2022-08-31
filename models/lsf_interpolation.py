"""
Created 12th July 2022

@author : minh.ngo
"""

import numpy as np
from scipy import ndimage
from scipy import interpolate
from astropy import wcs
import matplotlib.pyplot as plt
from astropy.io import fits
from ..lib.error import *
from . import LSF_DATA

class LSF_INTERPOLATION(object):
    """
    Interpolation by method, step of x-axis,
    step of y-axis
    """
    def __init__(self, list_lsf_data, listLines=None, method='linear', step_pos=1e-2, step_wave=10) -> None:
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
        method                  : string (default: 'linear')
                                method of interpolation
        step_pos                : float (default: 1e-2)
                                distance of step of relative position (Angstrom)
        step_wave               : float (default: 10)
                                step of wavelength of lines
        """            
        if listLines == None:
            self.list_lsf_data = np.asarray([list_lsf_data]) if type(list_lsf_data) == LSF_DATA else list_lsf_data
            li = np.empty(len(self.list_lsf_data), dtype=np.ndarray)
            for i in range(len(li)):
                li[i] = np.array(list(self.list_lsf_data[i].get_line_list().keys()))
            self._listLines = li
        else:
            if type(list_lsf_data) == LSF_DATA:
                self.list_lsf_data = np.asarray([list_lsf_data])
                self._listLines = np.asarray([[listLines]]) if type(listLines) == int else np.asarray([listLines]) 
            else:
                self.list_lsf_data = list_lsf_data
                self._listLines = listLines  
        for i in range(1, len(self.list_lsf_data)):
            if self.list_lsf_data[i] != self.list_lsf_data[0]:
                raise NameError("Config, detID or type de normalization did not work")
        for lsf in self.list_lsf_data:
            lsf.interp = True
        self._method = method
        self._step_pos = step_pos
        self._step_wave = step_wave
        self._interpolated_data = self._interpolate_data(self._method, self._step_pos, self._step_wave)


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

    def _interpolate_data(self, method, step_pos, step_wave):
        """ 
        Parameters
        -----------
        method      : str
                    method of interpolation 'nearest', 'linear', 'cubic'
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
        num = 0
        for li in self._listLines:
            num += len(li)
        if num <= 1:
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

    def plot_interpolate_data(self):
        """
        Visualize image after interpolating
        """
        data = self._interpolated_data
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

    def write_fits(self, filename):
        """
        Save image after interpolating the data

        Parameters
        -------------
        filename        : str
                        path to created file
        """
        data = self._interpolated_data
        grid_z = data['grid_z']
        with fits.open(self.file_arc) as hdul_arc:
            hdr = fits.Header()
            hdr = hdul_arc['PRIMARY'].header
            hdr['SLICE'] = self.slice
        empty_primary = fits.PrimaryHDU(header=hdr)
        coord = wcs.WCS(naxis=2)
        coord.wcs.crpix = np.array([1.0, 1.0])
        coord.wcs.crval = np.array([data['x'][0], data['y'][0]])
        coord.wcs.ctype = ['LINEAR', 'LINEAR']
        coord.wcs.cd = np.array([[self._step_pos, 0], [0, self._step_wave]])
        coord.wcs.set()
        image_hdu = fits.ImageHDU(grid_z, coord.to_header(), name='INTENSITY')
        hdul = fits.HDUList([empty_primary, image_hdu])
        hdul.writeto(filename, overwrite=True)

    def evaluate_intensity(self, w_0, waves):
        """
        Evaluation of intensity from a number of line in the slice
        take the closest wavelength of image to extract intensity

        Parameters
        ------------
        w_0     : float
                wavelength of lines
        waves   : float
                wavelength of each point in LSF 
                ATTENTION: not relative wavelength to w_0
        
        Returns
        -------------
        evaluated_intensity : array-like
                            intensity extracted from interpolated image
        """
        x_coor = [np.argmin(abs(self._interpolated_data['x']-w)) for w in waves-w_0]
        y_coor = np.argmin(abs(self._interpolated_data['y']-w_0))
        y_coor = np.full_like(x_coor, y_coor)
        evaluated_intensity = ndimage.map_coordinates(self._interpolated_data['grid_z'], [y_coor, x_coor], order=1)
        return evaluated_intensity

    def plot_evaluated_intensity(self, lsf_data: LSF_DATA, nb_line, ax, centre=True):
        """
        Visualization of intensity from a number of line in the slice
        take the closest wavelength of image to extract intensity

        Parameters
        ------------
        lsf_data    : LSF_DATA
                    data of one slice to measure
        nb_line     : int
                    number of line in the slice in lsf_data
        ax          : axes
                    ax to plot
        centre      : bool
                    True : relative wavelength to wavelength of line, which is centered in 0
                    False: wavelength of each point
        """        
        data = lsf_data.get_data_line(nb_line)
        waves = data['map_wave']
        w_0 = data['waveline']
        waves = np.linspace(min(waves), max(waves), len(waves))
        evaluated_intensity = self.evaluate_intensity(w_0, waves)
        if centre:
            ax.plot(waves-w_0, evaluated_intensity, label='Bspline data')
        else: 
            ax.plot(waves, evaluated_intensity, label='Bspline data')

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
            # Take evaluated intensity
            eval_intensity = self.evaluate_intensity(w_0, waves)
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
            listLines = np.arange(lsf_data._lineUp, lsf_data._lineDown+1)
        err = self.error_rms(lsf_data, listLines)
        wavelength_line = [lsf_data.get_data_line(nb_line)['waveline'] for nb_line in listLines]
        ax.plot(wavelength_line, err)
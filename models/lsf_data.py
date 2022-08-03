"""
Created 12th July 2022

@author : minh.ngo
"""

import hpylib as hp
import numpy as np
from astropy.io import fits
import math
from ..lib import normalize

class LSF_DATA:
    """
    Object contains all properties of a slice
    Configuration an lamp taken from a file_arc
    file_flat : correcting file
    """
    def __init__(self, file_arc: str, file_listLines: str, file_wavecal: str, file_slitlet: str,
                         slice=0, detID=1, normal=True, file_flat: str=None):
        """
        Constructor

        Parameters
        -----------
        file_arc        : str
                        path of a file arc
        file_listLines  : str
                        path to listLines (TXT or FITS file)
        file_wavecal    : str
                        path to a wavecal table
        file_slitlet    : str
                        path to slitlet table
        slice           : int
                        number of slice (0-37)
        detID           : int
                        number of detector (1-8)
        normal          : bool
                        normalized distribution of intensity
        file_flat       : str
                        path of file flat
        """
        self.file_arc = file_arc
        self.file_listLines = file_listLines
        self.file_wavecal = file_wavecal
        self.file_slitlet = file_slitlet
        self.file_flat = file_flat
        # Open image
        hdul = fits.open(file_arc)
        shape_image = hdul[f'CHIP{detID}.DATA'].shape
        if shape_image == (4096, 4096):
            self.pose = 'sampled'
        else:
            self.pose = 'oversampled'

        self.config = hdul['PRIMARY'].header['HIERARCH INM INS GRATING']
        self.slice = slice
        self.detID = detID
        self.normal = normal
        # Load line list
        status_lamp = [hdul['PRIMARY'].header[f'HIERARCH ESO FCS LAMP70 CHAN{i} STAT'] for i in range(1, 8)]
        try:
            indexOn = status_lamp.index('ON')
            lamp = hdul['PRIMARY'].header['HIERARCH ESO FCS LAMP70 CHAN{} NAME'.format(indexOn+1)]
            self.lamp = lamp.replace('LAMP_', '')
        except ValueError:
            self.lamp = 'linspace256'

        self._listLines = self.load_line_list(self.lamp, file_listLines)
        # Open table of wavelengths
        self._table_wave = hp.WAVECAL_TABLE.from_FITS(self.file_wavecal, self.detID)
        # Open slitlet table
        obj = hp.SLITLET_TABLE.from_FITS(self.file_slitlet, self.detID)
        
        ## Step of calibration 
        # Image (intensity)
        if file_flat != None:
            hdul_flat = fits.open(file_flat)
            self._image = hdul['CHIP'+str(self.detID)+'.DATA'].data/hdul_flat['CHIP'+str(self.detID)+'.DATA'].data
            hdul_flat.close() 
        else:
            self._image = hdul['CHIP'+str(self.detID)+'.DATA'].data
        hdul.close()
        
        # Get 3 columns of coordinate and wavelength left, center, right of the slice
        if self.pose == 'sampled':
            self._x_c = obj.get_xcenter(self.slice, y=np.arange(4096))
            self._x_r = obj.get_xright(self.slice, y=np.arange(4096))
            self._x_l = obj.get_xleft(self.slice, y=np.arange(4096))
            self._array_wave_r = self._table_wave.get_lbda(self.slice, x=self._x_r, y=np.arange(4096))
            self._array_wave_l = self._table_wave.get_lbda(self.slice, x=self._x_l, y=np.arange(4096))
            self._array_wave_c = self._table_wave.get_lbda(self.slice, x=self._x_c, y=np.arange(4096))
        else:
            self._x_c = obj.get_xcenter(self.slice, y=(np.arange(12288)-1)/3)*3+1
            self._x_r = obj.get_xright(self.slice, y=(np.arange(12288)-1)/3)*3+1
            self._x_l = obj.get_xleft(self.slice, y=(np.arange(12288)-1)/3)*3+1
            self._array_wave_r = self._table_wave.get_lbda(self.slice, x=(self._x_r-1)/3, y=(np.arange(12288)-1)/3)
            self._array_wave_l = self._table_wave.get_lbda(self.slice, x=(self._x_l-1)/3, y=(np.arange(12288)-1)/3) 
            self._array_wave_c = self._table_wave.get_lbda(self.slice, x=(self._x_c-1)/3, y=(np.arange(12288)-1)/3)
        mask = (np.min(self._array_wave_c) <= self._listLines) & (self._listLines <= np.max(self._array_wave_c))
        if len(self._listLines[mask]) == 0:
            raise NameError("EmptyLinesList")
        upLine = np.min(self._listLines[mask])
        downLine = np.max(self._listLines[mask])
        # lineUp, lineDown : first and last line in the slice
        self._lineUp = np.argmin(abs(self._listLines-upLine))
        self._lineDown = np.argmin(abs(self._listLines-downLine))
        self.pixel2dlambda = abs(np.nanmean(np.diff(self._array_wave_c)))


    def load_line_list(self, lamp, filename):
        """
        Get a list of wavelength from a TXT or FITS file

        Parameters
        ------------
        lamp        : str
                    a lamp ('Ar', 'Ne', 'Xe', 'Kr', 'linspace256')
        filename    : str
                    FITS or TXT file

        Returns
        -------------
        listLines   : array-like[int]
                    all number of lines of the lamp
        """
        if ".fits" in filename:
            hdul_lines = fits.open(filename)
            listLines = hdul_lines[self.config].data["wavelength"]
            hdul_lines.close()
        elif ".txt" in filename:
            listLines = np.genfromtxt(filename, usecols=1, skip_header=3)
        return listLines

    @classmethod
    def from_dict(obj, dic: dict):
        """
        Constructor from a dictionary
        """
        file_arc = dic['file_arc']
        file_listLines = dic['file_listLines']
        file_wavecal = dic['file_wavecal']
        file_slitlet = dic['file_slitlet']
        slice = dic['slice']
        detID = dic['detID']
        normal = dic['normal']
        file_flat = dic['file_flat']
        return obj(file_arc, file_listLines, file_wavecal, file_slitlet, slice, detID, normal, file_flat)

    def get_data_line(self, nb_line):
        """
        Extract all necessary informations for modelising the LSF

        Parameters
        ----------
        nb_line     : int
                    number of line in the slice

        Returns
        --------------
        dic     : dict['map_wave', 'waveline', 'intensity', 'x_coor', 'y_coor']
                necessary info
        """
        if not nb_line in range(self._lineUp, self._lineDown+1):
            raise NameError(f'This line is not in the slice {self.slice}')
        # Return 3 points left, center, right of the line
        wavelength_line = self._listLines[nb_line]
        ind = np.argmin(abs(self._array_wave_c - wavelength_line))
        point_c = (self._x_c[ind], ind)
        ind = np.argmin(abs(self._array_wave_r - wavelength_line))
        point_r = (self._x_r[ind], ind)
        ind = np.argmin(abs(self._array_wave_l - wavelength_line))
        point_l = (self._x_l[ind], ind)
    
        # Meshgrid, choose rectangle from coordinate x-y
        down_y = np.min([point_l[1], point_c[1], point_r[1]])
        upper_y = np.max([point_l[1], point_c[1], point_r[1]])
        if self.pose == 'sampled':
            # avoid y-coor being negative of greater than 4095
            if (down_y-4 < 0) & (upper_y+4+1 <= 4095):
                down_y_taken = 0
                upper_y_taken = upper_y+4+1
            elif (down_y-4 >= 0) & (upper_y+4+1 > 4095):
                down_y_taken = down_y-4
                upper_y_taken = 4095
            else:
                down_y_taken = down_y-4
                upper_y_taken = upper_y+4+1
            y_array = np.arange(down_y_taken, upper_y_taken)
            # Ignore 4 pixels from left and right bord
            x_array = np.arange(math.ceil(point_l[0])+4, math.floor(point_r[0])-4)
        else:
            if (down_y-12 < 0) & (upper_y+12+1 <= 12287):
                down_y_taken = 0
                upper_y_taken = upper_y+12+1
            elif (down_y-12 >= 0) & (upper_y+12+1 > 12287):
                down_y_taken = down_y-12
                upper_y_taken = 12287
            else:
                down_y_taken = down_y-12
                upper_y_taken = upper_y+12+1
            y_array = np.arange(down_y_taken, upper_y_taken)
            x_array = np.arange(math.ceil(point_l[0])+12, math.floor(point_r[0])-12)        
        x_cor, y_cor = np.meshgrid(x_array, y_array)

        # Image after being masked
        if self.pose == 'sampled':
            map_wave = self._table_wave.get_lbda(self.slice, x_cor, y_cor)
            mask = (abs(map_wave - wavelength_line) <= 4*self.pixel2dlambda)
        else:
            map_wave = self._table_wave.get_lbda(self.slice, (x_cor-1)/3, (y_cor-1)/3)
            mask = (abs(map_wave - wavelength_line) <= 12*self.pixel2dlambda)
        image_cut = self._image[y_cor, x_cor]        
        x_cor = x_cor[mask]
        y_cor = y_cor[mask]
        map_wave = map_wave[mask]
        image_cut = image_cut[mask]

        if self.normal:
            image_cut = normalize.normalize(image_cut)
        if self.pose == 'oversampled':
            x_cor = (x_cor-1)/3
            y_cor = (y_cor-1)/3
        dic = {'map_wave': map_wave, 'waveline': wavelength_line, 'intensity': image_cut, 'x_coor': x_cor,
                            'y_coor': y_cor}
        return dic
    
    def get_line_list(self):
        """
        Get all line from first and last line of the slice,
        ignore all lines that are close together
        Ex : {5: 15000, 6:15100, 9:16000}

        Returns
        ----------
        full_list : dict[indice, wavelength of line]
        """
        indices = range(self._lineUp, self._lineDown+1)
        full_list = dict(zip(indices, self._listLines[self._lineUp : self._lineDown+1]))
        y_lines = []
        # Get y-coordinate of all lines in the slice
        for nb_line in range(self._lineUp, self._lineDown+1):
            y_coor = self.get_data_line(nb_line)['y_coor']
            y_lines.append(np.mean(y_coor))
        diff_array = abs(np.diff(y_lines))
        # Ignore lines which are too close
        if self.pose == 'sampled':
            mask = (diff_array < 8*self.pixel2dlambda)
        else:
            mask = (diff_array < 24*self.pixel2dlambda)
        masked_diff_array = diff_array[mask]
        for elem in masked_diff_array:
            ind = np.argmin(abs(diff_array-elem))
            try:
                full_list.pop(self._lineUp+ind)
            except KeyError:
                pass
            try:
                full_list.pop(self._lineUp+ind+1)
            except KeyError:
                pass            
        return full_list


    def get_line_up(self):        
        """
        Get first line of the slice

        Returns
        ---------
        tup     : tuple[int, float]
                indice, wavelength of the first line of the slice
        """
        lines = self.get_line_list()
        first_key = list(lines.keys())[0]
        tup = first_key, lines[first_key]
        return tup

    def get_line_down(self):   
        """
        Get last line of the slice

        Returns
        ---------
        tup     : tuple[int, float]
                indice, wavelength of the last line of the slice
        """
        lines = self.get_line_list()
        last_key = list(lines.keys())[-1]
        tup = last_key, lines[last_key]
        return tup    
    
    def plot_line(self, nb_line, ax, centre=True):
        """
        Plot a line of intensity in function of wavelength of 
        the rectangle chosen

        Parameters
        -------------
        nb_line     : int
                    number of line (0-254)
        ax          : matplotlib.pyplot.axes

        """
        data = self.get_data_line(nb_line)
        if centre:
            ax.plot(data['map_wave']-data['waveline'], data['intensity'], '+')
        else:
            ax.plot(data['map_wave'], data['intensity'], '+')

    def scatter(self, nb_line, ax, centre = True, c: str = 'x_coor'):
        """
        Scatter with a colorbar
        """
        data = self.get_data_line(nb_line)
        if centre:
            sc = ax.scatter(data['map_wave']-data['waveline'], data['intensity'], c=data[c], marker='.', alpha=0.5, cmap='viridis')
        else:
            sc = ax.scatter(data['map_wave'], data['intensity'], c=data[c], marker='.', alpha=0.5, cmap='viridis')
        return sc

    def to_dict(self):
        """
        Parameters
        ------------

        Returns
        -------------
        dic : dict[file_arc, file_listLines, file_wavecal, file_slitlet, slice, detID, normal, file_flat]
        """
        dic = {'file_arc': self.file_arc, 'file_listLines': self.file_listLines, 'file_wavecal': self.file_wavecal,
                'file_slitlet': self.file_slitlet, 'slice': self.slice, 'detID': self.detID, 'normal': self.normal, 'file_flat': self.file_flat}
        return dic

    def __del__(self):
        """
        Destructor
        """
        pass
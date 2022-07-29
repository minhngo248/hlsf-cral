"""
Created on Thurs 7 July 2022

@author : minh.ngo
"""

import hpylib as hp
import numpy as np
from astropy.io import fits

def choose_line_max(pose, config, detID) -> int:
    """
    Parameters
    ------------
    pose        : string
                'sampled', 'oversampled'
    config      : string
                'H', 'HK', 'Hhigh'
    detID       : int
                1 to 8

    Returns
    -----------
    ind_chose   : int
                number of line with a maximum wavelength in the interval
    """
    hdul_rays = fits.open("../exposures/line_catalog_linspace256.fits")
    # Open table of wavelengths
    table_wave = hp.WAVECAL_TABLE.from_FITS("../exposures/WAVECAL_TABLE_20MAS_"+config+".fits", detID)
    # Open slitlet table
    obj = hp.SLITLET_TABLE.from_FITS("../exposures/SLITLET_TABLE_20MAS_"+config+".fits", detID)

    if pose == 'sampled':
        x_c_left = obj.get_xcenter(0, y=np.arange(4096))
        x_c_middle = obj.get_xcenter(19, y=np.arange(4096))
        x_c_right = obj.get_xcenter(37, y=np.arange(4096))
        wave_c_left = table_wave.get_lbda(0, x=x_c_left, y=np.arange(4096))
        wave_c_middle = table_wave.get_lbda(19, x=x_c_middle, y=np.arange(4096))
        wave_c_right = table_wave.get_lbda(37, x=x_c_right, y=np.arange(4096))        
    elif pose == 'oversampled':
        x_c_left = obj.get_xcenter(0, y=(np.arange(12288)-1)/3)*3+1
        x_c_middle = obj.get_xcenter(19, y=(np.arange(12288)-1)/3)*3+1
        x_c_right = obj.get_xcenter(37, y=(np.arange(12288)-1)/3)*3+1
        wave_c_left = table_wave.get_lbda(0, x=(x_c_left-1)/3, y=(np.arange(12288)-1)/3)
        wave_c_middle = table_wave.get_lbda(19, x=(x_c_middle-1)/3, y=(np.arange(12288)-1)/3)
        wave_c_right = table_wave.get_lbda(37, x=(x_c_right-1)/3, y=(np.arange(12288)-1)/3)
    
    listLine = np.array(hdul_rays[config].data["wavelength"])
    maxLine_left = np.max(listLine[(np.min(wave_c_left) <= listLine) & (listLine <= np.max(wave_c_left))])
    ind_left = np.argmin(abs(listLine - maxLine_left))
    maxLine_middle = np.max(listLine[(np.min(wave_c_middle) <= listLine) & (listLine <= np.max(wave_c_middle))])
    ind_middle = np.argmin(abs(listLine - maxLine_middle))
    maxLine_right = np.max(listLine[(np.min(wave_c_right) <= listLine) & (listLine <= np.max(wave_c_right))])
    ind_right = np.argmin(abs(listLine - maxLine_right))
    ind_chose = np.min([ind_left, ind_middle, ind_right]) - 1
    return ind_chose

def choose_line_min(pose, config, detID) -> int:
    """
    Parameters
    ------------
    pose :      string
                'sampled', 'oversampled'
    config :    string
                'H', 'HK', 'Hhigh'
    detID :     int
                1 to 8

    Returns
    -----------
    ind_chose   : int
                number of line with a minimum wavelength in the interval
    """
    hdul_rays = fits.open("../exposures/line_catalog_linspace256.fits")
    # Open table of wavelengths
    table_wave = hp.WAVECAL_TABLE.from_FITS("../exposures/WAVECAL_TABLE_20MAS_"+config+".fits", detID)
    # Open slitlet table
    obj = hp.SLITLET_TABLE.from_FITS("../exposures/SLITLET_TABLE_20MAS_"+config+".fits", detID)

    if pose == 'sampled':
        x_c_left = obj.get_xcenter(0, y=np.arange(4096))
        x_c_middle = obj.get_xcenter(19, y=np.arange(4096))
        x_c_right = obj.get_xcenter(37, y=np.arange(4096))
        wave_c_left = table_wave.get_lbda(0, x=x_c_left, y=np.arange(4096))
        wave_c_middle = table_wave.get_lbda(19, x=x_c_middle, y=np.arange(4096))
        wave_c_right = table_wave.get_lbda(37, x=x_c_right, y=np.arange(4096))        
    elif pose == 'oversampled':
        x_c_left = obj.get_xcenter(0, y=(np.arange(12288)-1)/3)*3+1
        x_c_middle = obj.get_xcenter(19, y=(np.arange(12288)-1)/3)*3+1
        x_c_right = obj.get_xcenter(37, y=(np.arange(12288)-1)/3)*3+1
        wave_c_left = table_wave.get_lbda(0, x=(x_c_left-1)/3, y=(np.arange(12288)-1)/3)
        wave_c_middle = table_wave.get_lbda(19, x=(x_c_middle-1)/3, y=(np.arange(12288)-1)/3)
        wave_c_right = table_wave.get_lbda(37, x=(x_c_right-1)/3, y=(np.arange(12288)-1)/3)
    
    listLine = np.array(hdul_rays[config].data["wavelength"])
    minLine_left = np.min(listLine[(np.min(wave_c_left) <= listLine) & (listLine <= np.max(wave_c_left))])
    ind_left = np.argmin(abs(listLine - minLine_left))
    minLine_middle = np.min(listLine[(np.min(wave_c_middle) <= listLine) & (listLine <= np.max(wave_c_middle))])
    ind_middle = np.argmin(abs(listLine - minLine_middle))
    minLine_right = np.min(listLine[(np.min(wave_c_right) <= listLine) & (listLine <= np.max(wave_c_right))])
    ind_right = np.argmin(abs(listLine - minLine_right))

    ind_chose = np.max([ind_left, ind_middle, ind_right]) + 1
    return ind_chose
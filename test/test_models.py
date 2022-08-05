"""
Test file for models

@author : minh.ngo
"""

import sys
import choose_line
from hlsf.models import *
import argparse
import matplotlib.pyplot as plt
import numpy as np


## Constants
lamps = ["Ar", "Kr", "Ne", "Xe"]
models_class = [GAUSSIAN_MODEL, MOFFAT_MODEL, GAUSS_HERMITE_MODEL]

def test_create_json(model, file_arc, file_listLines, file_wavecal, file_slitlet,
             slice, detID, file_flat=None, **kwargs):
    """
    Create JSON file

    Parameters
    ------------
    model           : class
                    GAUSSIAN_MODEL, MOFFAT_MODEL
    file_arc        : str
                    path of file arc to create model
    file_listLines  : str
                    path to listLines (../text/Ar.txt)
    file_wavecal    : str
                    path to wavecal table
    file_slitlet    : str
                    path to slitlet table
    slice           : int
                    slice of lsf_data
    detID           : int
                    number of detector
    filename_flat   : str
                    path of a flat file
    **kwargs        : 'deg' for GAUSS_HERMITE_MODEL
    """
    lsf_data = LSF_DATA(file_arc, file_listLines, file_wavecal, file_slitlet, slice, detID, file_flat=file_flat)
    try:
        deg = kwargs['deg']
    except KeyError:
        mod = model(lsf_data)
    else:
        mod = model(lsf_data, deg)
    mod.write_json(f"../file/{str.lower(model.__name__)}_{lsf_data.config}_{lsf_data.lamp}.json")

def test_evaluate_intensity(file_json, nb_line, config, slice):
    """
    Evaluate intensity of linspace256 from a model in JSON file

    Parameters
    ------------
    file_json       : str
                    path of JSON file
    nb_line         : int
                    number of line in the slice
    config          : str
                    'H', 'HK', 'Hhigh'
    slice           : int
                    0-37
    """
    mod = LSF_MODEL.from_json(file_json)
    lsf_data = LSF_DATA(f"../exposures/ARC-linspace256_CLEAR_20MAS_{config}_PRM.fits", "../exposures/line_catalog_linspace256.fits",
                                f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=slice)
    data = lsf_data.get_data_line(nb_line)
    w_0 = data['waveline']
    waves = data['map_wave']

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("wavelength")
    plt.ylabel("intensity")
    lsf_data.plot_line(nb_line, ax)
    mod.plot(w_0, waves, ax)
    ax.legend([f'Real data line {nb_line}', 'Fitted line'])
    plt.title(f'{str.lower(mod.__class__.__name__).replace("_"," ").capitalize()} RMS error {mod.lsf_data[0].lamp} {mod.error_rms(lsf_data, nb_line)}')
    plt.grid()
    plt.show()

def test_evaluate_delta(file_json, nb_line, config, slice):
    mod = LSF_MODEL.from_json(file_json)
    lsf_data = LSF_DATA(f"../exposures/ARC-linspace256_CLEAR_20MAS_{config}_PRM.fits", "../exposures/line_catalog_linspace256.fits",
                                f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=slice)
    data = lsf_data.get_data_line(nb_line)
    w_0 = data['waveline']
    delta_w = 4*lsf_data.pixel2dlambda

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("wavelength")
    plt.ylabel("intensity")
    lsf_data.plot_line(nb_line, ax)
    mod.plot_delta(w_0, delta_w, ax)
    ax.legend([f'Real data line {nb_line}', 'Fitted line'])
    plt.title(f'{str.lower(mod.__class__.__name__).replace("_"," ").capitalize()} RMS error {mod.lsf_data[0].lamp} {mod.error_rms(lsf_data, nb_line)}')
    plt.grid()
    plt.show()    

def test_rms_error(file_arc_test, file_lines, file_wavecal, file_slitlet, model, slice):
    """
    RMS error between 4 lamps Ar, Ne, Xe, Kr

    Parameters
    -------------
    file_arc_test   : str
                    path of evaluated file, normally linspace256
    file_lines      : str
    file_wavecal    : str
    file_slitlet    : str
    model           : type 
                    (ex: hlsf.GAUSSIAN_MODEL)
    slice           : int
    """
    lsf_data = LSF_DATA(file_arc_test, file_lines, file_wavecal, file_slitlet, slice=slice)
    mods = [model.from_json(f'../file/{str.lower(model.__name__)}_{lsf_data.config}_{lamp}.json') for lamp in lamps]
    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("wavelength")
    plt.ylabel("RMS error")
    for mod in mods:
        mod.plot_error_rms(lsf_data, range(lsf_data._lineUp, lsf_data._lineDown+1), ax)
    plt.legend(lamps)
    plt.title(f"Comparaison of {str.lower(model.__name__).replace('_',' ').capitalize()} of 4 lamps config {lsf_data.config}\nfrom line {lsf_data._lineUp} to line {lsf_data._lineDown}", fontweight='bold')
    plt.grid()
    plt.show()

def test_plot_parameters(model, lamp, config):
    """
    Plot parameters of model from JSON file

    Parameters
    -------------
    model       : class
    lamp        : str
    config      : str
    """
    name = str.lower(model.__name__)
    mod = LSF_MODEL.from_json(f"../file/{str.lower(model.__name__)}_{config}_{lamp}.json")
    fig, ax = plt.subplots(len(mod._dic_params), 1, figsize=(8, 16))
    plt.xlabel("wavelength")
    mod.plot_parameters(ax)
    plt.suptitle(f"{len(mod._dic_params)} parameters of {name.replace('_model', ' function').capitalize()}", fontweight='bold')
    plt.show()

def test_combine_lamps(model, config, slice):
    """
    Combination of 2 lamps, and import model in JSON file

    Parameters
    -------------
    model       : class
    config        : str
    slice      : str    
    """
    for i in range(len(lamps)):
        for j in range(i+1, len(lamps)):
            lsf_data = [LSF_DATA(f'../exposures/ARC-{lamps[i]}_CLEAR_20MAS_{config}_PRM.fits', f"../text/{lamps[i]}.txt", f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=slice), 
                LSF_DATA(f'../exposures/ARC-{lamps[j]}_CLEAR_20MAS_{config}_PRM.fits', f"../text/{lamps[j]}.txt", f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=slice)]
            mod = model(lsf_data)
            mod.write_json(f'../file/{str.lower(model.__name__)}_{config}_{lamps[i]}-{lamps[j]}.json')

def test_plot_9_recs(config, detID):
    """
    Plot 9 zones ou 9 rectangles

    Parameters
    ------------
    detID   : int
            1-8
    """
    slices = [0, 19, 37]
    obj = np.empty(3, dtype=LSF_DATA)
    for i, sli in enumerate(slices):
        obj[i] = LSF_DATA(f"../exposures/ARC-linspace256_CLEAR_20MAS_{config}_PRM.fits", "../exposures/line_catalog_linspace256.fits", 
                                f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=sli, detID=detID)
    max_line = choose_line.choose_line_max(obj[0].pose, obj[0].config, obj[0].detID)
    min_line = choose_line.choose_line_min(obj[0].pose, obj[0].config, obj[0].detID)
    lines = [min_line, 128, max_line]

    ## Plotting
    fig, axes = plt.subplots(3,3,figsize=(7,6)) 
    plt.xlabel('wavelength')
    plt.ylabel('intensity')
    # plot 3 upper rectangles
    for i in range(len(lines)):
        for j in range(len(slices)):
            obj[j].plot_line(lines[i], axes[i, j])
            axes[i, j].set_title(f" slice: {slices[j]} line: {lines[i]}")    
    plt.legend()
    plt.suptitle(f"LSF Variation 3 zones {obj[0].pose}, configuration {obj[0].config} detID {obj[0].detID}", fontsize=12, fontweight='bold')
    #plt.savefig(f"../images/3_zones_fitted_{obj[0].config}_detID_{obj[0].detID}")
    plt.show() 

def test_scatter_9_recs(config, detID):
    """
    Plot 9 zones ou 9 rectangles

    Parameters
    ------------
    detID   : int
            1-8
    """
    slices = [0, 19, 37]
    obj = np.empty(3, dtype=LSF_DATA)
    for i, sli in enumerate(slices):
        obj[i] = LSF_DATA(f"../exposures/ARC-linspace256_CLEAR_20MAS_{config}_PRM.fits", "../exposures/line_catalog_linspace256.fits", 
                                f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=sli, detID=detID)
    max_line = choose_line.choose_line_max(obj[0].pose, obj[0].config, obj[0].detID)
    min_line = choose_line.choose_line_min(obj[0].pose, obj[0].config, obj[0].detID)
    lines = [min_line, 128, max_line]

    ## Plotting
    fig, axes = plt.subplots(3,3,figsize=(7,6)) 
    plt.xlabel('wavelength')
    plt.ylabel('intensity')
    # plot 3 upper rectangles
    for i in range(len(lines)):
        for j in range(len(slices)):
            sc = obj[j].scatter(lines[i], axes[i, j])
            fig.colorbar(sc, ax=axes[i, j], label='x coordinate')
            axes[i, j].set_title(f" slice: {slices[j]} line: {lines[i]}")    
    plt.suptitle(f"LSF Variation 3 zones {obj[0].pose}, configuration {obj[0].config} detID {obj[0].detID}", fontsize=12, fontweight='bold')
    #plt.savefig(f"../images/3_zones_fitted_{obj[0].config}_detID_{obj[0].detID}")
    plt.show() 

def main() -> int:
    models_arg = ['G', 'M', 'GH']
    parser = argparse.ArgumentParser(description="Create models")
    parser.add_argument("-m", "--model", type=str, help="G: Gauss\nM: Moffat\nGH: Gauss-Hermite", choices=models_arg, default='G')
    parser.add_argument("--lamp", type=str, default='Ar')
    parser.add_argument("-c", "--config", type=str, help="Ex : H, HK, Hhigh", default='H')
    parser.add_argument("-s", "--slice", type=int, choices=range(38), default=0)
    parser.add_argument("-d", "--detID", type=int, choices=range(1,9), default=1)
    parser.add_argument("--nb_line", type=int, default=100)
    parser.add_argument("--deg", type=int, required='GH' in sys.argv, default=11)
    args = parser.parse_args()
    ind = models_arg.index(args.model)
    model = models_class[ind]
    lamp = args.lamp
    config = args.config
    slice = args.slice
    detID = args.detID
    nb_line = args.nb_line
    file_arc = f"../exposures/ARC-{lamp}_CLEAR_20MAS_{config}_PRM.fits"
    if lamp == "linspace256":
        file_listLines = "../exposures/line_catalog_linspace256.fits"
    else:
        file_listLines = f"../text/{lamp}.txt"
    file_wavecal = f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits"
    file_slitlet = f"../exposures/SLITLET_TABLE_20MAS_{config}.fits"
    file_flat = f"../exposures/FLAT-CONT2_CLEAR_20MAS_{config}_PRM.fits"

    print("Choose a test function")
    num = int(input('Enter a number (1-7): '))
    if num == 1:
        test_create_json(model, file_arc, file_listLines, file_wavecal, file_slitlet, slice, detID)
    elif num == 2:
        test_evaluate_intensity(f'../file/{str.lower(model.__name__)}_{config}_{lamp}.json', nb_line, config, slice)
    elif num == 3:
        test_rms_error(f"../exposures/ARC-{lamp}_CLEAR_20MAS_{config}_PRM.fits", file_listLines, file_wavecal, file_slitlet, model, slice) 
    elif num == 4:
        test_plot_parameters(model, lamp, config)
    elif num == 5:
        test_combine_lamps(model, config, slice)
    elif num == 6:
        test_plot_9_recs(config, detID)
    elif num == 7:
        test_scatter_9_recs(config, detID)
    elif num == 8:
        test_evaluate_delta(f'../file/{str.lower(model.__name__)}_{config}_{lamp}.json', nb_line, config, slice)
    return 0

main()
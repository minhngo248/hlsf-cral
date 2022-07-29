"""
Test file for models

@author : minh.ngo
"""

import sys
import hlsf
import argparse
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("..")
import choose_line

## Constants
lamps = ["Ar", "Kr", "Ne", "Xe"]
models_class = [hlsf.GAUSSIAN_MODEL, hlsf.MOFFAT_MODEL, hlsf.GAUSS_HERMITE_MODEL]

def test_create_json(model, filename_arc, file_listLines, slice, detID, filename_flat=None, **kwargs):
    """
    Create JSON file

    Parameters
    ------------
    model           : class
                    GAUSSIAN_MODEL, MOFFAT_MODEL
    filename_arc    : str
                    path of file arc to create model
    slice           : int
                    slice of lsf_data
    detID           : int
                    number of detector
    filename_flat   : str
                    path of a flat file
    **kwargs        : 'deg' for GAUSS_HERMITE_MODEL
    """
    lsf_data = hlsf.LSF_DATA(filename_arc, file_listLines, slice, detID, filename_flat=filename_flat)
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
    mod = hlsf.LSF_MODEL.from_json(file_json)
    lsf_data = hlsf.LSF_DATA(f"../exposures/ARC-linspace256_CLEAR_20MAS_{config}_PRM.fits", "../exposures/line_catalog_linspace256.fits", slice=slice)
    w_0 = lsf_data.get_data_line(nb_line)['waveline']
    waves = lsf_data.get_data_line(nb_line)['map_wave']

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("wavelength")
    plt.ylabel("intensity")
    lsf_data.plot_line(nb_line, ax)
    mod.plot(w_0, waves, ax)
    ax.legend([f'Real data line {nb_line}', 'Fitted line'])
    plt.title(f'RMS error {mod.lsf_data[0].lamp} {mod.error_rms(lsf_data, nb_line)}')
    plt.grid()
    plt.show()

def test_rms_error(file_arc_test, file_lines, model, slice):
    """
    RMS error between 4 lamps Ar, Ne, Xe, Kr

    Parameters
    -------------
    file_arc_test   : str
                    path of evaluated file, normally linspace256
    file_lines      : str
    model           : class
    slice           : int
    """
    lsf_data = hlsf.LSF_DATA(file_arc_test, file_lines, slice=slice)
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
    mod = hlsf.LSF_MODEL.from_json(f"../file/{str.lower(model.__name__)}_{config}_{lamp}.json")
    fig, ax = plt.subplots(len(mod._dic_params), 1, figsize=(8, 16))
    plt.xlabel("wavelength")
    mod.plot_parameters(ax)
    plt.suptitle(f"{len(mod._dic_params)} parameters of {name.replace('_model', ' function').capitalize()}", fontweight='bold')
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
    filename_arc = f"../exposures/ARC-{lamp}_CLEAR_20MAS_{config}_PRM.fits"
    if lamp == "linspace256":
        file_listLines = "../exposures/line_catalog_linspace256.fits"
    else:
        file_listLines = f"../text/{lamp}.txt"
    filename_flat = f"../exposures/FLAT-CONT2_CLEAR_20MAS_{config}_PRM.fits"

    print("Choose a test function")
    num = int(input('Enter a number (1-6): '))
    if num == 1:
        test_create_json(model, filename_arc, file_listLines, slice, detID, filename_flat)
    elif num == 2:
        test_evaluate_intensity('../file/gaussian_model_H_Ar.json', nb_line, config, slice)
    elif num == 3:
        test_rms_error("../exposures/ARC-linspace256_CLEAR_20MAS_H_PRM.fits", "../exposures/line_catalog_linspace256.fits", model, slice) 
    elif num == 4:
        test_plot_parameters(model, lamp, config)
    return 0

main()
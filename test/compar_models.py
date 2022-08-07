"""
Comparison between 3 models

@author : minh.ngo
"""

from hlsf.models import *
import matplotlib.pyplot as plt
import numpy as np
import argparse

def test_rms_models(config, slice):
    """
    RMS error 3 models
    
    Parameters
    -------------
    config      : str
    slice       : int
                0-37
    """
    lsf_data = LSF_DATA(f"../exposures/ARC-linspace256_CLEAR_20MAS_{config}_PRM.fits", "../exposures/line_catalog_linspace256.fits",
                            f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=slice)
    mod = MOFFAT_MODEL.from_json("../file/moffat_model_H_Ne.json")
    mod1 = GAUSSIAN_MODEL.from_json("../file/gaussian_model_H_Ne.json")
    mod2 = GAUSS_HERMITE_MODEL.from_json("../file/gauss_hermite_model_H_Ne.json")

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("wavelength")
    plt.ylabel("rms error")
    mod.plot_error_rms(lsf_data, ax)
    mod1.plot_error_rms(lsf_data, ax)
    mod2.plot_error_rms(lsf_data, ax)
    ax.legend(['Moffat', 'Gauss', f'GH deg {mod2.deg}'])
    plt.title(f"RMS error between 3 models from line {lsf_data._lineUp} to {lsf_data._lineDown}", fontweight='bold')
    plt.grid()
    plt.show()

def test_plot(config, nb_line, slice):
    """
    Plot evaluated lines
    
    Parameters
    -------------
    config      : str
    nb_line     : int
    slice       : int
                0-37
    """
    lsf_data = LSF_DATA(f"../exposures/ARC-linspace256_CLEAR_20MAS_{config}_PRM.fits", "../exposures/line_catalog_linspace256.fits",
                            f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=slice)
    mod = np.empty(3, dtype=object)
    mod[0] = MOFFAT_MODEL.from_json("../file/moffat_model_H_Ne.json")
    mod[1] = GAUSSIAN_MODEL.from_json("../file/gaussian_model_H_Ne.json")
    mod[2] = GAUSS_HERMITE_MODEL.from_json("../file/gauss_hermite_model_H_Ne.json")

    list_string = [f'{mod[i].__class__.__name__} {mod[i].error_rms(lsf_data, nb_line)}\n' for i in range(len(mod))]
    string = ''
    for str in list_string:
        string += str 
    data = lsf_data.get_data_line(nb_line)
    waves = data['map_wave']
    w_0 = data['waveline']

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("wavelength")
    plt.ylabel("intensity")
    lsf_data.plot_line(nb_line, ax)
    for m in mod:
        m.plot(w_0, waves, ax)
    plt.grid()
    plt.legend(['Real data', 'Moffat', 'Gauss', f'Gauss Hermite deg {mod[2].deg}'])
    plt.title(f"RMS error {string}")
    plt.show()

def test_2_models_rms(model: str, lamp, slice):
    mods = [LSF_MODEL.from_json(f'../file/{model}_H_{lamp}.json'), LSF_MODEL.from_json(f'../file/{model}_2_H_{lamp}.json')]
    lsf_data = LSF_DATA(f"../exposures/ARC-linspace256_CLEAR_20MAS_H_PRM.fits", "../exposures/line_catalog_linspace256.fits",
                            f"../exposures/WAVECAL_TABLE_20MAS_H.fits", f"../exposures/SLITLET_TABLE_20MAS_H.fits", slice=slice)
    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("wavelength")
    plt.ylabel("RMS error")
    mods[0].plot_error_rms(lsf_data, ax)
    mods[1].plot_error_rms(lsf_data, ax)
    plt.legend(['Model 1', 'Model 2'])
    plt.title(f"RMS error {model.replace('_',' ').capitalize()}")
    plt.grid()
    plt.show()

def plot_parameters(model: str, lamp):
    mods = [LSF_MODEL.from_json(f'../file/{model}_H_{lamp}.json'), LSF_MODEL.from_json(f'../file/{model}_2_H_{lamp}.json')]
    if model == "gaussian_model":
        fig, axes = plt.subplots(3, 1)
        mods[0].plot_parameters(axes)
        mods[1].plot_parameters(axes)
    elif model == "gauss_hermite_model":
        fig, axes = plt.subplots(3, 4)
        mods[0].plot_parameters(axes, (3,4))
        mods[1].plot_parameters(axes, (3,4))   
    plt.xlabel("wavelength")
    fig.suptitle(f"{model.replace('_',' ').capitalize()}") 
    plt.legend()
    plt.show()    



def main() -> int:
    parser = argparse.ArgumentParser(description="Create models")
    parser.add_argument("-m", "--model", type=str, help="Ex : G, GH", choices=['G','GH'], default='G')
    parser.add_argument("-c", "--config", type=str, help="Ex : H, HK, Hhigh", default='H')
    parser.add_argument("-s", "--slice", type=int, choices=range(38), default=0)
    parser.add_argument("--nb_line", type=int, default=100)
    parser.add_argument("-l", "--lamp", type=str, default='Ar')
    args = parser.parse_args()
    lamp = args.lamp
    config = args.config
    slice = args.slice
    nb_line = args.nb_line
    if args.model == 'G':
        model = "gaussian_model"
    else:
        model = "gauss_hermite_model"

    print("Choose a test function")
    num = int(input('Enter a number (1-4): '))
    if num == 1:
        test_rms_models(config, slice)
    elif num == 2:
        test_plot(config, nb_line, slice)
    elif num == 3:
        test_2_models_rms(model, lamp, slice)
    elif num == 4:
        plot_parameters(model, lamp)
    return 0

main()

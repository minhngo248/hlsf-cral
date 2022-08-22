from hlsf.models import *
import matplotlib.pyplot as plt
import argparse
from hlsf.models import LSF_INTERPOLATION

lamps = ["Ar", "Kr", "Ne", "Xe"]

def test_plot_interpolate(file_arc, file_listLines, file_wavecal, file_slitlet, slice):
    """
    Visualization of interpolated image
    """
    lsf_data = LSF_DATA(file_arc, file_listLines, file_wavecal, file_slitlet, slice)
    lsf_data.plot_interpolate_data(method='linear')

def test_evaluate_intensity(nb_line, config, slice):
    """
    Evaluate intensity of linspace256 from a model in JSON file
    Gaussian model 2 et interpolation 

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
    mod = LSF_MODEL.from_json(f'../file/gaussian_model_2_{config}_Ar.json')
    lsf_data = LSF_DATA(f"../exposures/ARC-linspace256_CLEAR_20MAS_{config}_PRM.fits", "../exposures/line_catalog_linspace256.fits",
                                f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=slice)
    data = lsf_data.get_data_line(nb_line)
    w_0 = data['waveline']
    waves = data['map_wave']
    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel(r"wavelength ($\AA$)")
    plt.ylabel("intensity")
    lsf_data.plot_line(nb_line, ax)
    mod.plot(w_0, waves, ax)
    lsf_data.plot_evaluated_intensity(nb_line, ax)
    plt.legend(['Real data', 'Gaussian model', 'Interpolation'])
    plt.title(f'Line {nb_line}')
    plt.show()

def interpolation_lamps(config, slice):
    list_lsf_data = [LSF_DATA(f"../exposures/ARC-{lamp}_CLEAR_20MAS_{config}_PRM.fits", f"../text/{lamp}.txt",
                                f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=slice) for lamp in lamps]
    lsf_interp = LSF_INTERPOLATION(list_lsf_data)
    lsf_interp.plot_interpolate_data()

def interpolation_lamps_intensity(file_arc, file_listLines, file_wavecal, file_slitlet, slice, nb_line):
    lsf_data = LSF_DATA(file_arc, file_listLines, file_wavecal, file_slitlet, slice=slice)
    list_lsf_data = [LSF_DATA(f"../exposures/ARC-{lamp}_CLEAR_20MAS_{lsf_data.config}_PRM.fits", f"../text/{lamp}.txt",
                                f"../exposures/WAVECAL_TABLE_20MAS_{lsf_data.config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{lsf_data.config}.fits", slice=slice) for lamp in lamps]
    lsf_interp = LSF_INTERPOLATION(list_lsf_data)
    ax = plt.axes()
    lsf_data.plot_line(nb_line, ax)
    lsf_interp.plot_evaluated_intensity(lsf_data, nb_line, ax)
    plt.legend(['Real data', 'interpolated data'])
    plt.title(f'RMS error {lsf_interp.error_rms(lsf_data, nb_line)}')
    plt.grid()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Create models")
    parser.add_argument("--lamp", type=str, default='Ar')
    parser.add_argument("-c", "--config", type=str, help="Ex : H, HK, Hhigh", default='H')
    parser.add_argument("-s", "--slice", type=int, choices=range(38), default=0)
    parser.add_argument("--nb_line", type=int, default=100)
    args = parser.parse_args()
    lamp = args.lamp
    config = args.config
    slice = args.slice
    nb_line = args.nb_line
    file_arc = f"../exposures/ARC-{lamp}_CLEAR_20MAS_{config}_PRM.fits"
    file_listLines = f"../text/{lamp}.txt"
    file_wavecal = f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits"
    file_slitlet = f"../exposures/SLITLET_TABLE_20MAS_{config}.fits"
    num = int(input('Type a number (1-4): '))
    if num == 1:
        test_plot_interpolate(file_arc, file_listLines, file_wavecal, file_slitlet, slice)
    elif num == 2:
        test_evaluate_intensity(nb_line, config, slice)
    elif num == 3:
        interpolation_lamps(config, slice)
    elif num == 4:
        interpolation_lamps_intensity(f'../exposures/ARC-linspace256_CLEAR_20MAS_{config}_PRM.fits', '../exposures/line_catalog_linspace256.fits', 
                file_wavecal, file_slitlet, slice, nb_line)

main()
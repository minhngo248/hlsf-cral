"""
3 models and lsf_interpolation
"""
from hlsf.models import *
import matplotlib.pyplot as plt
import argparse

def test_rms_models(config, slice, lamp):
    """
    RMS error 3 models : GAUSSIAN_MODEL_2, MOFFAT_MODEL_2
    GAUSS_HERMITE_MODEL_2 and LSF_INTERPOLATION
    
    Parameters
    -------------
    config      : str
    slice       : int
                0-37
    lamp        : str
                'Ar', 'Kr', 'Ne', 'Xe'
    """
    lsf_data = LSF_DATA(f"../exposures/ARC-linspace256_CLEAR_20MAS_{config}_PRM.fits", "../exposures/line_catalog_linspace256.fits",
                            f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=slice)
    mod = MOFFAT_MODEL_2.from_json(f"../file/moffat_model_2_H_{lamp}.json")
    mod1 = GAUSSIAN_MODEL_2.from_json(f"../file/gaussian_model_2_H_{lamp}.json")
    mod2 = GAUSS_HERMITE_MODEL_2.from_json(f"../file/gauss_hermite_model_2_H_{lamp}.json")
    lsf = LSF_DATA(f"../exposures/ARC-{lamp}_CLEAR_20MAS_{config}_PRM.fits", f"../text/{lamp}.txt",
                            f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits", f"../exposures/SLITLET_TABLE_20MAS_{config}.fits", slice=slice)
    lsf_interp = LSF_INTERPOLATION(lsf)
    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel("wavelength")
    plt.ylabel("rms error")
    mod.plot_error_rms(lsf_data, ax)
    mod1.plot_error_rms(lsf_data, ax)
    mod2.plot_error_rms(lsf_data, ax)
    lsf_interp.plot_error_rms(lsf_data, ax)
    ax.legend(['Moffat', 'Gauss', f'GH deg {mod2.deg}', 'Interpolation'])
    plt.title(f"RMS error between 3 models from line {lsf_data._lineUp} to {lsf_data._lineDown} lamp {lamp}", fontweight='bold')
    plt.grid()
    plt.savefig(f"../images/compar_and_interp_GH_deg_{mod2.deg}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Comparison of models")
    parser.add_argument("-c", "--config", type=str, help="Ex : H, HK, Hhigh", default='H')
    parser.add_argument("-s", "--slice", type=int, choices=range(38), default=0)
    parser.add_argument("-l", "--lamp", type=str, default='Ar')
    args = parser.parse_args()
    lamp = args.lamp
    config = args.config
    slice = args.slice
    print('1. Show RMS error of 3 models\n')    
    num = int(input("Choose a number: "))
    if num == 1:
        test_rms_models(config, slice, lamp)

main()
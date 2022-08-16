from hlsf.models import LSF_DATA
import argparse

def test_plot_interpolate(file_arc, file_listLines, file_wavecal, file_slitlet, slice):
    lsf_data = LSF_DATA(file_arc, file_listLines, file_wavecal, file_slitlet, slice)
    lsf_data.plot_interpolate_data(method='cubic', step=1)

def main():
    parser = argparse.ArgumentParser(description="Create models")
    parser.add_argument("--lamp", type=str, default='Ar')
    parser.add_argument("-c", "--config", type=str, help="Ex : H, HK, Hhigh", default='H')
    parser.add_argument("-s", "--slice", type=int, choices=range(38), default=0)
    args = parser.parse_args()
    lamp = args.lamp
    config = args.config
    slice = args.slice
    file_arc = f"../exposures/ARC-{lamp}_CLEAR_20MAS_{config}_PRM.fits"
    file_listLines = f"../text/{lamp}.txt"
    file_wavecal = f"../exposures/WAVECAL_TABLE_20MAS_{config}.fits"
    file_slitlet = f"../exposures/SLITLET_TABLE_20MAS_{config}.fits"
    test_plot_interpolate(file_arc, file_listLines, file_wavecal, file_slitlet, slice)

main()
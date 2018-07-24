#!/Users/ding/anaconda3/bin/python
# Copied the code from nersc. --06/14/2018
# -----------------------------------------------------------------------------
# Combine multiple binary files with data Cijl and Gm into a whole binary data file. In this way, we do not need to modify code
# which calculates Cijl' and G'. --06/12/2018
#
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Combine binary data files of Cijl ang Gm, made by Zhejie.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lt", help = '*The type of weak lensing survey. TF: Tully-Fisher; TW: traditional weak lensing.', required=True)
parser.add_argument("--nzbin", help = '*The number of tomographic bins.', required=True)
parser.add_argument("--nkbin_in", help = 'Number of k bins of input power spectrum.', type=int, default=505)
parser.add_argument("--nkbin_out", help = '*Number of output k bins.', type=int, required=True)
parser.add_argument("--nrank", help = "The number of ranks used to generate individual binary data files.", type=int, default=4)
parser.add_argument("--Pk_type", help = '*The type of power spectrum, either Pwig_nonlinear or Pnow.', required=True)
parser.add_argument("--survey_stage", help = '*Survey_stage, e.g., KW_stage_IV/precise_Cijl_Gm', required=True)
parser.add_argument("--get_cijl", help = 'Whether to obtain Cijl data, either True of others.')
parser.add_argument("--get_gm", help = 'Whether to obtain G matrix data, either True of others.')

args = parser.parse_args()
lt = args.lt
nzbin = args.nzbin
nkbin_in = args.nkbin_in
nkbin_out = args.nkbin_out
nrank = args.nrank
Pk_type = args.Pk_type
survey_stage = args.survey_stage

def main():
    prefix_dict = {'TF': 'Tully-Fisher', 'TW': 'TW_zext'}

    idir0 = './{}/mpi_preliminary_data_{}/'.format(survey_stage, Pk_type)
    idir1 = 'comm_size{}/'.format(nrank)
    idir = idir0 + idir1

    odir0 = idir0

    def combine_data(file_rank, nkbin_, ofile):
        fwriter = open(ofile, 'ab')
        for rank_id in range(nrank):
            ifile = idir + file_rank.format(prefix_dict[lt], nzbin, nkbin_, rank_id)
            data = np.fromfile(ifile, dtype=np.float64, count=-1, sep="")
            data.tofile(fwriter, sep="")

        fwriter.close()

    def combine_Cijl():
        if lt == 'TF':
            ofile = odir0 + '{}_Cij_l_{}rbins_{}kbins_CAMB.bin'.format(prefix_dict[lt], nzbin, nkbin_in)
            file_rank = '{}_Cij_l_{}rbins_{}kbins_CAMB_rank{}.bin'
        elif lt == 'TW':
            ofile = odir0 + '{}_Cij_l_{}rbins_ext_{}kbins_CAMB.bin'.format(prefix_dict[lt], nzbin, nkbin_in)
            file_rank = '{}_Cij_l_{}rbins_ext_{}kbins_CAMB_rank{}.bin'
        fwriter = open(ofile, 'wb')
        fwriter.close()

        combine_data(file_rank, nkbin_in, ofile)

    def combine_Gmatrix():
        ofile = odir0 + '{}_Gm_cross_out_{}rbins_{}kbins_CAMB.bin'.format(prefix_dict[lt], nzbin, nkbin_out)
        file_rank = '{}_Gm_cross_out_{}rbins_{}kbins_CAMB_rank{}.bin'
        fwriter = open(ofile, 'wb')  # if the file doesn't exist, create it; otherwise erase it.
        fwriter.close()

        combine_data(file_rank, nkbin_out, ofile)

    if args.get_cijl == 'True':
        combine_Cijl()
    if args.get_gm == 'True':
        combine_Gmatrix()

if __name__ == '__main__':
    main()

#!/Users/ding/miniconda3/bin/python
# Copy it from the code mpi_PW_sn_dep_Cov_multibin.py in folder PW_stage_IV. The difference from KW_SN_ratio_Takada_Jain.py is the range of
# tomographic bins in the input file. Note that what I output is (S/N)^2 instead of (S/N) --06/04/2018
#
from mpi4py import MPI
import numpy as np
import math
import os, sys
from scipy import integrate
from scipy import linalg
from functools import reduce
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/TW_f2py_SVD/')
from cov_matrix_module import cal_cov_matrix
import argparse

#---------- set parameters correspondingly ------------#
#snf = 5.93e-5        # this matches the scale factor 1.e-3 in Tully-Fisher case
#---------------------------------------------------------
parser = argparse.ArgumentParser(description="Calculate signal-to-noise ratio (square) for C^ij(l), made by Zhejie.")
parser.add_argument("--snf", help = '*The shape noise factor from the default value.', required=True)
parser.add_argument("--nbin_case", help = '*Case id for the number of tomographic bins. See the directory num_rbin in the code.', type=int, required=True)
parser.add_argument("--Pk_type", help = "*The type of input P(k), whether it's linear (Pwig), or damped (Pwig_nonlinear), or without BAO (Pnow).", required=True)
parser.add_argument("--comm_size", help = '*The number of the processes used to generate preliminary data.', required=True)
parser.add_argument("--idir0", help = "*The basic directory of input files, e.g., './precise_Cijl_Gm/'.", required=True)
parser.add_argument("--odir0", help = "*The basic directory of output files, e.g., './precise_Cijl_Gm/'.", required=True)

args = parser.parse_args()
snf = float(args.snf)
nbin_case = args.nbin_case
Pk_type = args.Pk_type
comm_size = args.comm_size
idir0 = args.idir0
odir0 = args.odir0

#---------------------------------------------------------
def cal_signal_noise_ratio():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()
    t_start = MPI.Wtime()

    #num_rbin = {'num_sbin': (5, 15, 30, 100, 150), 'num_pbin': (6, 19, 38, 127, 191)}    # s: spectroscopic; p: photometric -- This case is correct only for n(z) Stage-III.
    num_rbin = {'num_sbin': (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 22, 25, 27, 30, 32, 35, 37),
                'num_pbin': (2, 3, 5, 6, 7, 8, 10, 11, 12, 18, 25, 27, 31, 34, 37, 40, 44, 46)}    # s: spectroscopic; p: photometric

    red_bin = num_rbin['num_sbin'][nbin_case]
    red_bin_ext = num_rbin['num_pbin'][nbin_case]
    N_dset = (red_bin+1)*red_bin//2
    N_dset_ext = (red_bin_ext+1)*red_bin_ext//2

    num_kin = 505
    l_min = 1
    l_max = 2002
    #delta_l = 1
    delta_l = 3
    num_l = (l_max - l_min)//delta_l + 1
    f_sky = 15000.0/41253.0            # Survey area 15000 deg^2 is from PW-Stage IV (LSST)
    #print("f_sky: ", f_sky)
    data_type_size = 8

    prefix = 'TW_zext_'
    idir = idir0 + 'mpi_preliminary_data_{}/comm_size{}/'.format(Pk_type, comm_size)
    #------------- !! write output files, they are the basic files --------------#
    ofdir = odir0 + 'mpi_{}sn_exp_k_data_{}/comm_size{}/signal_noise_ratio/'.format(prefix, Pk_type, size)
    ofprefix = ofdir + prefix
    print('Output file prefix:', ofprefix)
    if rank == 0:
        if not os.path.exists(ofdir):
            os.makedirs(ofdir)
        # read shape noise term \sigma^2/n^i
        inputf = idir0 + 'mpi_preliminary_data_Pwig_nonlinear/comm_size{}/'.format(comm_size) + prefix + 'pseudo_shapenoise_{0}rbins_ext.out'.format(red_bin_ext)
        pseudo_sn_ext = np.loadtxt(inputf, dtype='f8', comments='#')
        pseudo_sn = np.array(pseudo_sn_ext[0: red_bin])*snf
        print(pseudo_sn.shape)
    else:
        pseudo_sn = np.zeros(red_bin)
    comm.Bcast(pseudo_sn, root=0)

    default_num_l_in_rank = int(np.ceil(num_l / size))
    # Rounding errors here should not be a problem unless default size is very small
    end_num_l_in_rank = num_l - (default_num_l_in_rank * (size - 1))
    assert end_num_l_in_rank >= 1, "Assign fewer number of processes."

    if (rank == (size - 1)):
        num_l_in_rank = end_num_l_in_rank
    else:
        num_l_in_rank = default_num_l_in_rank

    # be careful here we have extended photometric redshift bins, which is different from TF case.
    Cijl_len = num_l_in_rank * N_dset_ext
    Cijl_sets = np.zeros(Cijl_len)

    # default case with delta_l = 3
    file_Cijl_cross = idir + prefix + 'Cij_l_{}rbins_ext_{}kbins_CAMB_rank{}.bin'.format(red_bin_ext, num_kin, rank) # Cij_l stores Cij for each ell by row
    Cijl_freader = open(file_Cijl_cross, 'rb') # Open and read a binary file
    Cijl_sets = np.fromfile(Cijl_freader, dtype='d', count=-1, sep='')
    #print('Cij(l) from rank', rank, 'is:', Cijl_sets, '\n')
    comm.Barrier()
    Cijl_freader.close()

    def SNR_fun(l, rank):
        n_l = default_num_l_in_rank * rank + l
        ell = l_min + n_l * delta_l
        # put the whole array cij for an given ell into the upper triangle part of the matrix
        cijl_array = Cijl_sets[l*N_dset_ext: (l+1)*N_dset_ext]
        #print(cijl_array, cijl_array.shape)
        cijl_m[iu1] = np.array(cijl_array)
        #print(cijl_m, cijl_m.shape)
        cijl_m_select = np.array(cijl_m[0:red_bin, 0:red_bin])   # select the first red_bin bins of Cij, match the case with T-F
        cijl_true = np.array(cijl_m_select[iu2])                   # convert upper triangle matrix to an array
        cijl_sn = np.array(cijl_true)
        cijl_sn[sn_id] = cijl_true[sn_id] + pseudo_sn              # add shape noise terms in Cii(l) terms

        Cov_cij_cpq = cal_cov_matrix(red_bin, iu2, cijl_sn)        # calculate the covariance matrix of Cij(l), Cpq(l')
        Cov_cij_cpq = Cov_cij_cpq.T + np.triu(Cov_cij_cpq, k=1)    # It's symmetric. From up triangular matrix, construct the whole matrix for inversion.

        #inv_Cov_cij_cpq = linalg.inv(Cov_cij_cpq, overwrite_a=True)
        inv_Cov_cij_cpq = linalg.inv(Cov_cij_cpq)

        inv_Cov_cij_cpq = inv_Cov_cij_cpq * ((2.0*ell+1.0)*delta_l*f_sky)             # Take account of the number of modes (the denominator)
        SN_per_ell = reduce(np.dot, [cijl_true, inv_Cov_cij_cpq, cijl_true])
        if rank == size-1:
            print('ell from rank', rank, 'is', ell, '\n')
            #print(np.allclose(np.dot(Cov_cij_cpq, inv_Cov_cij_cpq), np.eye(N_dset)))
        return ell, SN_per_ell

    #-------- Output signal-to-noise ratio from each ell -------##
    SN_ofile = ofprefix + 'SNR_per_ell_{}rbins_{}kbins_snf{}_rank{}.dat'.format(red_bin, num_kin, snf, rank)
    SN_square = np.array([], dtype=np.float64).reshape(0, 2)

    iu1 = np.triu_indices(red_bin_ext)
    iu2 = np.triu_indices(red_bin)

    sn_id = [(2*red_bin+1-ii)*ii//2 for ii in range(red_bin)]  # id of dset C^ij(l) which is added with shot noise
    cijl_m = np.zeros((red_bin_ext, red_bin_ext))               # one matrix to store C^ij at one ell
    for l in range(num_l_in_rank):
        ell, SN_per_ell = SNR_fun(l, rank)
        SN_square = np.vstack((SN_square, np.array([ell, SN_per_ell])))
    header_line = " ell      (S/N)^2"
    np.savetxt(SN_ofile, SN_square, fmt='%.7e', delimiter=' ', newline='\n', header=header_line, comments='#')
    comm.Barrier()

    t_end = MPI.Wtime()
    if rank == 0:
        print('With total processes', size, ', the running time:', t_end-t_start)

def main():
    cal_signal_noise_ratio()

if __name__ == '__main__':
    main()

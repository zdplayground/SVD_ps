#!/Users/ding/miniconda3/bin/python
# Copy code from mpi_TF_sn_dep_Cov_cij_cross_bin.py, modify it to append data after the uncomplete output files. -- 11/27/2017
# -------------------------------------------------------------------------------------------------------------------------------------
# ----- This code is for the local computer environment. -- 10/25/2017
# Copy the code ../../PW_stage_IV/nersc_code/mpi_PW_sn_dep_Cov_Cij_bin.py, modify it for kinematic weak lensing case. -- 10/25/2017
# Set the option (hard coding, change idir) to read preliminary Pwig_nonlinear data with 66 kout bins from ../TF_cross-ps. -- 11/02/2017
# Remember that before running the code read the caption first! I forgot changing the directory of Cij_l and shapenoise file. -- 11/26/2017
# Still need to double check the code. -- 11/27/2017
#
from mpi4py import MPI
import numpy as np
import os, sys
from scipy import integrate
from scipy import linalg
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/TF_cross-ps/')
from cov_matrix_module import cal_cov_matrix
import argparse

#---------------------------------------------------------
parser = argparse.ArgumentParser(description="Calculate G' with shape noise dependence in Cov(C^ii(l), C^jj(l)), made by Zhejie.")
parser.add_argument("--nrbin", help = 'Number of tomographic bins.', type=int, required=True)
parser.add_argument("--num_kout", help = '*Number of output k bins.', type=int, required=True)
parser.add_argument("--snf", help = '*The shape noise factor from the default value.', type=float, required=True)
parser.add_argument("--Pk_type", help = "*The type of input P(k), whether it's linear (Pwig), or damped (Pwig_nonlinear), or without BAO (Pnow).", required=True)
parser.add_argument("--odir0", help = "*The basic directory of output files, e.g., './' for the current directory.", required=True)
parser.add_argument("--num_size", help = "The total number of processes used in MPI.", type=int, required=True)

args = parser.parse_args()
num_rbin = args.nrbin
num_kout = args.num_kout
snf = args.snf
Pk_type = args.Pk_type
odir0 = args.odir0
num_size = args.num_size

#---------------------------------------------------------
def cal_sn_dep_Cov_cij():
    N_dset = (num_rbin+1)*num_rbin//2
    num_kin = 505
    l_min = 1
    l_max = 2002
    delta_l = 3
    num_l = (l_max - l_min)//delta_l + 1
    f_sky = 15000.0/41253.0            # Survey area 15000 deg^2 is from TF-Stage IV
    #print("f_sky: ", f_sky)
    data_type_size = 8

    prefix = 'Tully-Fisher_'
    idir = odir0 + 'mpi_preliminary_data_{}/'.format(Pk_type)
    #------------- !! write output files, they are the basic files --------------#
    ofdir = odir0 + 'mpi_{}sn_exp_k_data_{}/comm_size{}/'.format(prefix, Pk_type, num_size)
    ifprefix = idir + prefix
    ofprefix = ofdir + prefix
    #print('Output file prefix:', ofprefix)

    inputf = ifprefix + 'pseudo_shapenoise_{0}rbins.out'.format(num_rbin)   # read the default shape noise \sigma^2/n^i
    pseudo_sn = np.loadtxt(inputf, dtype='f8', comments='#')
    pseudo_sn = pseudo_sn * snf                                                  # times a shape noise scale factor

    # default case with delta_l = 3
    file_Cijl_cross = ifprefix + 'Cij_l_{0}rbins_{1}kbins_CAMB.bin'.format(num_rbin, num_kin) # Cij_l stores Cij for each ell by row
    Cijl_freader = open(file_Cijl_cross, 'rb')

    #--------------- !! read Gm_cross part by part for each ell -----------------#
    file_Gm_cross = odir0 + 'mpi_preliminary_data_Pwig_nonlinear/' + prefix + 'Gm_cross_out_{0}rbins_{1}kbins_CAMB.bin'.format(num_rbin, num_kout)
    Gm_freader = open(file_Gm_cross, 'rb')
    iu1 = np.triu_indices(num_rbin)
    sn_id = [int((2*num_rbin+1-ii)*ii/2) for ii in range(num_rbin)]          # The ID of dset C^ij(l) added by the shape noise when i=j (auto power components)

    def cal_C_G(l):
        ell = l_min + delta_l * l                              # l is from 0
        # put the whole array cij at ell into the upper triangle part of the matrix
        cijl_true = np.fromfile(Cijl_freader, dtype='d', count=N_dset, sep='')   # read in the true C^ij(l) for each l case
        cijl_sn = np.array(cijl_true)                                 # Distinguish the observed C^ijl) (denoted by cijl_sn) from the true C^ij(l)
        cijl_sn[sn_id] = cijl_true[sn_id] + pseudo_sn                 # Add shape noise on C^ii terms to get cijl_sn
        # if rank == 0:
        #     print('cijl_sn:', cijl_sn)
        Cov_cij_cpq = cal_cov_matrix(num_rbin, iu1, cijl_sn)           # Get an upper-triangle matrix for Cov(C^ij, C^pq) from Fortran subroutine wrapped.
        Cov_cij_cpq = Cov_cij_cpq/((2.0*ell+1.0)*delta_l*f_sky)             # Take account of the number of modes (the denominator)
        # if rank == 0:
        #     print('Cov_cij_cpq:', Cov_cij_cpq)
        w_ccij, v_ccij = linalg.eigh(Cov_cij_cpq, lower=False, overwrite_a=True)  # Get eigenvalue and eigenvectors from Scipy routine
        w_inv = 1./w_ccij
        # if rank == 0:
        #     print('w_ccij', w_ccij, 'v_ccij:', v_ccij)
        #--If uncomment the below, overwrite_a should be set False in the linalg.eigh()
        # sqrt_w_inv = np.diag(w_inv**0.5)
        # v_inv = np.transpose(v_ccij)
        # Cov_cij_sym = np.triu(Cov_cij_cpq, k=1) + Cov_cij_cpq.T      # Once Cov_cij_cpq set to be overwrited, the criterion couldn't be used anymore
        # print(reduce(np.dot, [np.diag(w_inv), v_inv, Cov_cij_sym, v_inv.T]))

        Cov_inv_half = np.transpose(w_inv**0.5 *v_ccij)               # Simplify the expression of dot(sqrt_w_inv, v_inv), 05/09/2016
        # if rank == 0:
        #     print('Cov_inv_half:', Cov_inv_half)
        G_l_array = np.fromfile(Gm_freader, dtype='d', count=N_dset*num_kout, sep='')

        Gm_l = np.reshape(G_l_array, (N_dset, num_kout), 'C')         # In Python, the default storage of a matrix follows C language format.
        #print Gm_l[0,:]
        Gm_l = np.dot(Cov_inv_half, Gm_l)
        cijl_true = np.dot(Cov_inv_half, cijl_true)
        return cijl_true, Gm_l


    default_num_l_in_rank = int(np.ceil(num_l/num_size))
    l_temp = 0
    rank_list = np.arange(1, 4)
    n_ell_list = [15, 8, 4]
    count = 0
    for rank in rank_list:
        #------------- !! Gm_prime output
        Gm_prime_file = ofprefix + 'Gm_cross_prime_{}rbins_{}kbins_snf{}_rank{}.bin'.format(num_rbin, num_kout, snf, rank)
        print(Gm_prime_file)
        Gm_prime_fwriter = open(Gm_prime_file, 'ab')
        #-------- generate C^ij(l) prime -------##
        Cijl_prime_file = ofprefix + 'Cijlprime_{}rbins_{}kbins_snf{}_rank{}.bin'.format(num_rbin, num_kin, snf, rank)
        print(Cijl_prime_file)
        Cijl_prime_fwriter = open(Cijl_prime_file, 'ab')

        n_ell_infile = n_ell_list[count]    # here the explicit number represents # of ells have been already written from nersc code.
        l_start = rank * default_num_l_in_rank + n_ell_infile
        for i in range(l_temp, l_start):
            cijl_array = np.fromfile(Cijl_freader, dtype='d', count=N_dset, sep='')   # read in the true C^ij(l) for each l case
            G_l_array = np.fromfile(Gm_freader, dtype='d', count=N_dset*num_kout, sep='')

        del cijl_array
        del G_l_array
        if rank < num_size-1:
            l_end = (rank+1) * default_num_l_in_rank
        elif rank == num_size-1:
            l_end = num_l
        for l in range(l_start, l_end):
            print('l:', l)
            cijl_true, Gm_l = cal_C_G(l)
            cijl_true.tofile(Cijl_prime_fwriter, sep="")
            Gm_l.tofile(Gm_prime_fwriter, sep="")

        Gm_prime_fwriter.close()
        Cijl_prime_fwriter.close()
        l_temp = l_end
        count = count + 1

    Cijl_freader.close()
    Gm_freader.close()

def main():
    cal_sn_dep_Cov_cij()

if __name__ == '__main__':
    main()

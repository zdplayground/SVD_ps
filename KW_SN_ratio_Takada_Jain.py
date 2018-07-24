#!/Users/ding/anaconda3/bin/python
# Copy the code from mpi_TF_sn_dep_Cov_cij_cross_bin.py in KW_stage_IV, modify it to calculate signal-to-noise ratio shown in Takada & Jain 2004(or 2008). --06/01/2018
# Add the prefix "KW" on the code name to distinguish it from the case PW. The difference is due to additional content of C^ijl(l) (tomographic bins) in PW case. --06/02/2018
# Find that what I output is (S/N)^2 instead of (S/N). --06/04/2018
#
from mpi4py import MPI
import numpy as np
import os, sys
from scipy import integrate
from scipy import linalg
from functools import reduce
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/TF_cross-ps/')
from cov_matrix_module import cal_cov_matrix
import argparse

#---------------------------------------------------------
parser = argparse.ArgumentParser(description="Calculate G' with shape noise dependence in Cov(C^ii(l), C^jj(l)), made by Zhejie.")
parser.add_argument("--nrbin", help = 'Number of tomographic bins.', type=int, required=True)
parser.add_argument("--num_kout", help = '*Number of output k bins.', type=int, required=True)
parser.add_argument("--snf", help = '*The shape noise factor from the default value.', type=float, required=True)
parser.add_argument("--Pk_type", help = "*The type of input P(k), whether it's linear (Pwig), or damped (Pwig_nonlinear), or without BAO (Pnow).", required=True)
parser.add_argument("--Psm_type", help = '*The expression of Pnorm. The default case, Pnorm from Eisenstein & Zaldarriaga 1999. \
                                          If Pnorm=Pnow, it is derived from transfer function.')
parser.add_argument("--idir0", help = "*The basic directory of input files, e.g., './precise_Cijl_Gm/'.", required=True)
parser.add_argument("--odir0", help = "*The basic directory of output files, e.g., './precise_Cijl_Gm/'.", required=True)

args = parser.parse_args()
num_rbin = args.nrbin
num_kout = args.num_kout
snf = args.snf
Pk_type = args.Pk_type
Psm_type = args.Psm_type
idir0 = args.idir0
odir0 = args.odir0

#---------------------------------------------------------
def cal_signal_noise_ratio():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()
    t_start = MPI.Wtime()

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
    idir = idir0 + 'mpi_preliminary_data_{}/'.format(Pk_type)
    #------------- !! write output files, they are the basic files --------------#
    ofdir = odir0 + 'mpi_{}sn_exp_k_data_{}/comm_size{}/signal_noise_ratio/'.format(prefix, Pk_type, size)
    ifprefix = idir + prefix
    if Psm_type == 'Pnow':
        Gm_ifprefix = idir0 + 'mpi_preliminary_data_Pwig_nonlinear/set_Pnorm_Pnow/' + prefix
        ofdir = ofdir + 'set_Pnorm_Pnow/'
    else:
        Gm_ifprefix = idir0 + 'mpi_preliminary_data_Pwig_nonlinear/' + prefix
    ofprefix = ofdir + prefix
    #print('Output file prefix:', ofprefix)
    if rank == 0:
        if not os.path.exists(ofdir):
            os.makedirs(ofdir)

        inputf = Gm_ifprefix + 'pseudo_shapenoise_{0}rbins.out'.format(num_rbin)   # read the default shape noise \sigma^2/n^i
        print(inputf)
        pseudo_sn = np.loadtxt(inputf, dtype='f8', comments='#')
        pseudo_sn = pseudo_sn * snf                                                  # times a shape noise scale factor
        #print(pseudo_sn.shape)
    else:
        pseudo_sn = np.zeros(num_rbin)
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
    Cijl_len = num_l_in_rank * N_dset
    Cijl_sets = np.zeros(Cijl_len)

    # default case with delta_l = 3
    file_Cijl_cross = ifprefix + 'Cij_l_{0}rbins_{1}kbins_CAMB.bin'.format(num_rbin, num_kin) # Cij_l stores Cij for each ell by row
    Cijl_freader = MPI.File.Open(comm, file_Cijl_cross) # Open and read a binary file
    Cijl_fh_start = rank * Cijl_len * data_type_size    # need to calculate how many bytes shifted
    Cijl_freader.Seek(Cijl_fh_start)
    Cijl_freader.Read([Cijl_sets, MPI.DOUBLE])          # Read using individual file pointer
    #print('Cij(l) from rank', rank, 'is:', Cijl_sets, '\n')
    comm.Barrier()
    Cijl_freader.Close()

    def SNR_fun(l, rank):
        n_l = default_num_l_in_rank * rank + l
        ell = l_min + n_l * delta_l
        cijl_true = Cijl_sets[l*N_dset: (l+1)*N_dset]
        # if rank == 0:
        #     print(cijl_true)
        cijl_sn = np.array(cijl_true)                                  # Distinguish the observed C^ijl) (denoted by cijl_sn) from the true C^ij(l)
        cijl_sn[sn_id] = cijl_true[sn_id] + pseudo_sn                  # Add shape noise on C^ii terms to get cijl_sn
        # if rank == 0:
        #     print('cijl_sn:', cijl_sn)
        Cov_cij_cpq = cal_cov_matrix(num_rbin, iu1, cijl_sn)           # Get an upper-triangle matrix for Cov(C^ij, C^pq) from Fortran subroutine wrapped.
        Cov_cij_cpq = Cov_cij_cpq.T + np.triu(Cov_cij_cpq, k=1)        # It's symmetric. Construct the whole matrix for inversion.
        inv_Cov_cij_cpq = linalg.inv(Cov_cij_cpq, )
        inv_Cov_cij_cpq = inv_Cov_cij_cpq * ((2.0*ell+1.0)*delta_l*f_sky)             # Take account of the number of modes (the denominator)
        SN_per_ell = reduce(np.dot, [cijl_true, inv_Cov_cij_cpq, cijl_true])

        if rank == size-1:
            print('ell from rank', rank, 'is', ell, '\n')
        return ell, SN_per_ell

    #-------- Output signal-to-noise ratio from each ell -------##
    SN_ofile = ofprefix + 'SNR_per_ell_{}rbins_{}kbins_snf{}_rank{}.dat'.format(num_rbin, num_kin, snf, rank)
    SN_square = np.array([], dtype=np.float64).reshape(0, 2)

    iu1 = np.triu_indices(num_rbin)
    sn_id = [int((2*num_rbin+1-ii)*ii/2) for ii in range(num_rbin)]          # The ID of dset C^ij(l) added by the shape noise when i=j (auto power components)
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

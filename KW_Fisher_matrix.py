#!/Users/ding/anaconda3/bin/python
# Copy the code from KW_sn_ratio_Takada_Jain.py, modify it to calculate the Fisher matrix Cijl_wig-Cijl_now, reference is taken from, e.g. Eq. (26) in Hu & Jain 2004.
# It seems that we don't need to consider shape noise in the covariance matrix of shear power spectra difference between Cijl_wig and Cijl_now. --07/25/2018
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
parser = argparse.ArgumentParser(description="Calculate Fisher matrix of Cijl_wig-Cijl_now, made by Zhejie.", )
parser.add_argument("--nrbin", help = 'Number of tomographic bins.', type=int, required=True)
parser.add_argument("--snf", help = '*The shape noise factor from the default value.', type=float, required=True)
parser.add_argument("--Pk_type", help = "*The type of input P(k), whether it's linear (Pwig), or damped (Pwig_nonlinear), or without BAO (Pnow).", required=True)
parser.add_argument("--Psm_type", help = '*The expression of Pnorm. The default case, Pnorm from Eisenstein & Zaldarriaga 1999. \
                                          If Pnorm=Pnow, it is derived from transfer function.')
parser.add_argument("--idir0", help = "*The basic directory of input files, e.g., './precise_Cijl_Gm/'.", required=True)
parser.add_argument("--odir0", help = "*The basic directory of output files, e.g., './precise_Cijl_Gm/'.", required=True)
parser.add_argument("--alpha", help = "The BAO scale shifting parameter alpha, i.e. k'=alpha * k or l'= alpha * l given a \chi. Value could be e.g. 1.02.", required=True, type=np.float64)

args = parser.parse_args()
num_rbin = args.nrbin
snf = args.snf
Pk_type = args.Pk_type
Psm_type = args.Psm_type
idir0 = args.idir0
alpha = args.alpha

#---------------------------------------------------------
def cal_Fisher():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()
    odir0 = args.odir0
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
    idir_default = idir0 + 'mpi_preliminary_data_{}/'.format(Pk_type)
    idir_shift = idir0 + 'BAO_alpha_{}/mpi_preliminary_data_{}/'.format(alpha, Pk_type)
    ifprefix_default = idir_default + prefix
    ifprefix_shift = idir_shift + prefix

    #------------- !! write output files, they are the basic files --------------#
    ofdir = odir0 + 'BAO_alpha_{}/mpi_{}sn_exp_k_data_{}/comm_size{}/'.format(alpha, prefix, Pk_type, size)
    ofprefix = ofdir + prefix

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
    Cijl_sets_default = np.zeros(Cijl_len)
    Cijl_sets_shift = np.zeros(Cijl_len)

    # Read the default Cijl, with delta_l = 3
    file_Cijl_cross = ifprefix_default + 'Cij_l_{0}rbins_{1}kbins_CAMB.bin'.format(num_rbin, num_kin) # Cij_l stores Cij for each ell by row
    Cijl_freader = MPI.File.Open(comm, file_Cijl_cross) # Open and read a binary file
    Cijl_fh_start = rank * Cijl_len * data_type_size    # need to calculate how many bytes shifted
    Cijl_freader.Seek(Cijl_fh_start)
    Cijl_freader.Read([Cijl_sets_default, MPI.DOUBLE])          # Read using individual file pointer
    #print('Cij(l) from rank', rank, 'is:', Cijl_sets, '\n')
    comm.Barrier()
    Cijl_freader.Close()
    # Read the Cijl with shift ell value
    file_Cijl_cross = ifprefix_shift + 'Cij_l_{0}rbins_{1}kbins_CAMB.bin'.format(num_rbin, num_kin) # Cij_l stores Cij for each ell by row
    Cijl_freader = MPI.File.Open(comm, file_Cijl_cross) # Open and read a binary file
    Cijl_fh_start = rank * Cijl_len * data_type_size    # need to calculate how many bytes shifted
    Cijl_freader.Seek(Cijl_fh_start)
    Cijl_freader.Read([Cijl_sets_shift, MPI.DOUBLE])          # Read using individual file pointer
    #print('Cij(l) from rank', rank, 'is:', Cijl_sets, '\n')
    comm.Barrier()
    Cijl_freader.Close()

    def Fisher_element(l, rank):
        n_l = default_num_l_in_rank * rank + l
        ell = l_min + n_l * delta_l
        cijl_default = Cijl_sets_default[l*N_dset: (l+1)*N_dset]
        cijl_shift = Cijl_sets_shift[l*N_dset: (l+1)*N_dset]
        cijl_diff = cijl_shift - cijl_default

        Cov_cij_cpq = cal_cov_matrix(num_rbin, iu1, cijl_diff)           # Get an upper-triangle matrix for Cov(C^ij, C^pq) from Fortran subroutine wrapped.
        Cov_cij_cpq = Cov_cij_cpq.T + np.triu(Cov_cij_cpq, k=1)          # It's symmetric. Construct the whole matrix for inversion.
        inv_Cov_cij_cpq = linalg.inv(Cov_cij_cpq, overwrite_a=True)
        inv_Cov_cij_cpq = inv_Cov_cij_cpq * ((2.0*ell+1.0)*delta_l*f_sky)             # Take account of the number of modes (the denominator) and f_sky
        dcijl_dalpha = cijl_diff/(alpha-1.0)
        faa = reduce(np.dot, [dcijl_dalpha, inv_Cov_cij_cpq, dcijl_dalpha])

        # if rank == size-1:
        #     print('ell from rank', rank, 'is', ell, '\n')
        return ell, faa

    #-------- Output signal-to-noise ratio from each ell -------##
    # faa_ofile = ofprefix + 'SNR_per_ell_{}rbins_{}kbins_snf{}_rank{}.dat'.format(num_rbin, num_kin, snf, rank)
    # SN_square = np.array([], dtype=np.float64).reshape(0, 2)

    iu1 = np.triu_indices(num_rbin)
    #sn_id = [int((2*num_rbin+1-ii)*ii/2) for ii in range(num_rbin)]          # The ID of dset C^ij(l) added by the shape noise when i=j (auto power components)
    Faa = 0.0
    for l in range(num_l_in_rank):
        ell, faa = Fisher_element(l, rank)
        Faa = Faa + faa
    print('Faa:', Faa, 'from rank:', rank)
    # header_line = " ell      (S/N)^2"
    # np.savetxt(SN_ofile, SN_square, fmt='%.7e', delimiter=' ', newline='\n', header=header_line, comments='#')

    comm.reduce(Faa, op=MPI.SUM, root=0)
    comm.Barrier()
    t_end = MPI.Wtime()
    if rank == 0:
        print('The Fisher element Faa is:', Faa)
        print('With total processes', size, ', the running time:', t_end-t_start)

def main():
    cal_Fisher()

if __name__ == '__main__':
    main()

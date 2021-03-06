#!/Users/ding/anaconda3/bin/python
# Copy code from ../KW_stage_IV, modify f_sky for KW_stage_III. -- 09/21/2018
# ---------------------------------------------------------------------------
# Add the optional parameter Psm_type which is calculated from Pnorm or Pnow (transfer function). -- 05/21/2018
# Remember that before running the code read the caption first! I forgot changing the directory of Cij_l and shapenoise file. -- 11/26/2017
# 1. I don't know why the eigenvalue of Cov_cij_cpq from "now" case becomes negative when shape noise factor is less than 0.05. Is that because that transfer function
# deviates from the broadband shape of power spectrum a little bit? We may test it by using Zvonimir's power spectrum for the preliminary Cijl. -- 08/08/2018
# 2. To use Cijl data file from ./ and G matrix from ./precise_Cijl_Gm/, we specifiy input directory of Cijl and Gm. --08/08/2018
from mpi4py import MPI
import numpy as np
import os, sys
from scipy import integrate
from scipy import linalg
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/common_modules/')
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
parser.add_argument("--idir0_Cijl", help = "*The basic input directory of Cijl, e.g., ./BAO_alpha_1.0/.", required=True)
parser.add_argument("--idir0_Gm", help = "*The basic input directory of Gm, e.g., ./BAO_alpha_1.0/.", required=True)
parser.add_argument("--odir0", help = "*The basic directory of output files, e.g., ./BAO_alpha_1.0/.", required=True)

args = parser.parse_args()
num_rbin = args.nrbin
num_kout = args.num_kout
snf = args.snf
Pk_type = args.Pk_type
Psm_type = args.Psm_type
idir0_Cijl = args.idir0_Cijl
idir0_Gm = args.idir0_Gm

#---------------------------------------------------------
def cal_sn_dep_Cov_cij():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()
    t_start = MPI.Wtime()

    odir0 = args.odir0
    N_dset = (num_rbin+1)*num_rbin//2
    num_kin = 505
    l_min = 1
    l_max = 2002
    delta_l = 3
    num_l = (l_max - l_min)//delta_l + 1
    f_sky = 5000.0/41253.0            # Survey area 5000 deg^2 is from TF-Stage III
    #print("f_sky: ", f_sky)
    data_type_size = 8

    prefix = 'Tully-Fisher_'
    idir = idir0_Cijl + 'mpi_preliminary_data_{}/'.format(Pk_type)
    #------------- !! write output files, they are the basic files --------------#
    ofdir = odir0 + 'mpi_{}sn_exp_k_data_{}/comm_size{}/'.format(prefix, Pk_type, size)
    ifprefix = idir + prefix
    if Psm_type == 'Pnow':
        Gm_ifprefix = idir0_Gm + 'mpi_preliminary_data_Pwig_nonlinear/set_Pnorm_Pnow/' + prefix
        ofdir = ofdir + 'set_Pnorm_Pnow/'
    else:
        Gm_ifprefix = idir0_Gm + 'mpi_preliminary_data_Pwig_nonlinear/' + prefix
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
    Gm_len = num_l_in_rank * N_dset * num_kout
    Gm_sets = np.zeros(Gm_len)

    # default case with delta_l = 3
    file_Cijl_cross = ifprefix + 'Cij_l_{0}rbins_{1}kbins_CAMB.bin'.format(num_rbin, num_kin) # Cij_l stores Cij for each ell by row
    Cijl_freader = MPI.File.Open(comm, file_Cijl_cross) # Open and read a binary file
    Cijl_fh_start = rank * Cijl_len * data_type_size    # need to calculate how many bytes shifted
    Cijl_freader.Seek(Cijl_fh_start)
    Cijl_freader.Read([Cijl_sets, MPI.DOUBLE])          # Read using individual file pointer
    #print('Cij(l) from rank', rank, 'is:', Cijl_sets, '\n')
    comm.Barrier()
    Cijl_freader.Close()

    #--------------- !! read Gm_cross part by part for each ell -----------------#
    file_Gm_cross = Gm_ifprefix + 'Gm_cross_out_{0}rbins_{1}kbins_CAMB.bin'.format(num_rbin, num_kout)
    Gm_freader = MPI.File.Open(comm, file_Gm_cross)
    Gm_fh_start = rank * Gm_len * data_type_size
    Gm_freader.Seek(Gm_fh_start)
    Gm_freader.Read([Gm_sets, MPI.DOUBLE])
    #print('Gm from rank', rank, 'is:', Gm_sets.shape, '\n')
    comm.Barrier()
    Gm_freader.Close()

    def cal_C_G(l, rank):
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
        if rank == 0:
            rank_matrix = np.linalg.matrix_rank(Cov_cij_cpq)
            print('ell, rank of Cov:', ell, rank_matrix)
        #     ofile = ofprefix + "Cov_cij_cpq_ell_{}_{}rbins_{}kbins_snf{}_rank{}.npz".format(ell, num_rbin, num_kin, snf, rank)
        #     np.savez_compressed(ofile, Cov_cij_cpq = Cov_cij_cpq)
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
        # Cov_cij_sym = np.triu(Cov_cij_cpq, k=1) + Cov_cij_cpq.T      # Once Cov_cij_cpq set to be overwriten, the criterion couldn't be used anymore
        # print(reduce(np.dot, [np.diag(w_inv), v_inv, Cov_cij_sym, v_inv.T]))

        Cov_inv_half = np.transpose(w_inv**0.5 *v_ccij)               # Simplify the expression of dot(sqrt_w_inv, v_inv), 05/09/2016
        # if rank == 0:
        #     print('Cov_inv_half:', Cov_inv_half)
        G_l_array = Gm_sets[l*N_dset*num_kout: (l+1)*N_dset*num_kout]

        Gm_l = np.reshape(G_l_array, (N_dset, num_kout), 'C')         # In Python, the default storage of a matrix follows C language format.
        #print Gm_l[0,:]
        Gm_l = np.dot(Cov_inv_half, Gm_l)

        cijl_true = np.dot(Cov_inv_half, cijl_true)
        # if rank == 0:
        #     print('ell from rank', rank, 'is', ell, '\n')
        return cijl_true, Gm_l


    amode = MPI.MODE_WRONLY|MPI.MODE_CREATE

    #-------- generate C^ij(l) prime -------##
    Cijl_prime_file = ofprefix + 'Cijlprime_{}rbins_{}kbins_snf{}_rank{}.bin'.format(num_rbin, num_kin, snf, rank)
    Cijl_prime_fwriter = open(Cijl_prime_file, 'wb')

    #------------- !! Gm_prime output
    Gm_prime_file = ofprefix + 'Gm_cross_prime_{}rbins_{}kbins_snf{}_rank{}.bin'.format(num_rbin, num_kout, snf, rank)
    Gm_prime_fwriter = open(Gm_prime_file, 'wb')

    iu1 = np.triu_indices(num_rbin)
    sn_id = [int((2*num_rbin+1-ii)*ii/2) for ii in range(num_rbin)]          # The ID of dset C^ij(l) added by the shape noise when i=j (auto power components)
    for l in range(num_l_in_rank):
        cijl_true, Gm_l = cal_C_G(l, rank)
        cijl_true.tofile(Cijl_prime_fwriter, sep="")
        Gm_l.tofile(Gm_prime_fwriter, sep="")
        del cijl_true, Gm_l

    comm.Barrier()
    Cijl_prime_fwriter.close()
    Gm_prime_fwriter.close()
    t_end = MPI.Wtime()
    if rank == 0:
        print('shape noise factor:', snf)
        print('With total processes', size, ', the running time:', t_end-t_start)

def main():
    cal_sn_dep_Cov_cij()

if __name__ == '__main__':
    main()

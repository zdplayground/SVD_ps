#!/Users/ding/anaconda3/bin/python
# Copy the code from ../PW_stage_IV, modify f_sky,
# -------------------------------------------------------------------
# 1. It's better to output data into separate files from each process. Because it's faster than collectively writing data into one
# single file (7 times faster with 4 processes running). Also it doesn't introduce systematics in the matrix inversion process. --10/22/2017
# 2. Modify reading in Cijl and G data for the case of G with 66 output k bins. -- 06/15/2018
# 3. For generating G' with 66 output k bins, comment out the output of Cijl'. -- 06/16/2018
# 4. Output of Cijl' and G' with shape noise factor 0.1838. --07/06/2018
# 5. Be careful about inputting Cijl and G data files; whether we read a single file or multiple files with rank id appended. --07/13/2018
# 6. Add the additional parameter alpha denoting the BAO scale shifting parameter. It's useful to do Fisher analysis. --09/03/2018
#
from mpi4py import MPI
import numpy as np
import math
import os, sys
from scipy import integrate
from scipy import linalg
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/common_modules/')
from cov_matrix_module import cal_cov_matrix
import argparse

#---------------------------------------------------------
parser = argparse.ArgumentParser(description="Calculate Cijl' and G' with shape noise dependence in Cov(C^ii(l), C^jj(l)), made by Zhejie.", formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--snf", help = '*The shape noise factor from the default value.', required=True)
parser.add_argument("--nbin_case", help = '*Case id for the number of tomographic bins. See the directory num_rbin in the code.', type=int, required=True)
parser.add_argument("--num_kout", help = '*Number of output k bins.', type=int, required=True)
parser.add_argument("--Pk_type", help = "*The type of input P(k), whether it's linear (Pwig), or damped (Pwig_nonlinear), or without BAO (Pnow).", required=True)
parser.add_argument("--comm_size", help = '*The number of the processes used to generate preliminary data.', required=True)
parser.add_argument("--alpha", help = 'alpha, the BAO scale shifting parameter.', default=1.0, type=np.float, required=True)

args = parser.parse_args()
snf = float(args.snf)
nbin_case = args.nbin_case
num_kout = args.num_kout
Pk_type = args.Pk_type
comm_size = args.comm_size
alpha = args.alpha

#---------------------------------------------------------
def cal_sn_dep_Cov_cij():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()
    t_start = MPI.Wtime()

    num_rbin = {'num_sbin': (5, 30),
                'num_pbin': (6, 38)}    # s: spectroscopic; p: photometric

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
    f_sky = 5000.0/41253.0            # Survey area 5000 deg^2 is for PW-Stage III (DES)
    #print("f_sky: ", f_sky)
    data_type_size = 8

    prefix = 'TW_zext_'
    idir0 = './BAO_alpha_{}/'.format(alpha)
    ##idir = './mpi_preliminary_data_{}/comm_size{}/'.format(Pk_type, comm_size)
    idir = idir0 + 'mpi_preliminary_data_{}/'.format(Pk_type)
    #------------- !! write output files, they are the basic files --------------#
    ofdir = idir0 + 'mpi_{}sn_exp_k_data_{}/comm_size{}/'.format(prefix, Pk_type, size)
    Gm_ifprefix = idir0 + 'mpi_preliminary_data_Pwig_nonlinear/' + prefix
    ofprefix = ofdir + prefix
    print('Output file prefix:', ofprefix)
    if rank == 0:
        if not os.path.exists(ofdir):
            os.makedirs(ofdir)
        # read shape noise term \sigma^2/n^i
        inputf = Gm_ifprefix + 'pseudo_shapenoise_{0}rbins_ext.out'.format(red_bin_ext)
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
    Gm_len = num_l_in_rank * N_dset_ext * num_kout
    Gm_sets = np.zeros(Gm_len)

    # default case with delta_l = 3
    file_Cijl_cross = idir + prefix + 'Cij_l_{}rbins_ext_{}kbins_CAMB.bin'.format(red_bin_ext, num_kin) # Cij_l stores Cij for each ell by row
    Cijl_freader = MPI.File.Open(comm, file_Cijl_cross) # Open and read a binary file
    Cijl_fh_start = rank * Cijl_len * data_type_size    # need to calculate how many bytes shifted
    Cijl_freader.Seek(Cijl_fh_start)
    Cijl_freader.Read([Cijl_sets, MPI.DOUBLE])          # Read using individual file pointer
    #print('Cij(l) from rank', rank, 'is:', Cijl_sets, '\n')
    comm.Barrier()
    Cijl_freader.Close()
    # Since Cijl has been generated with equal number of ells from different 4 ranks, we could directly read data from sub-binary files.
    # file_Cijl_cross = idir + 'comm_size{}/'.format(comm.size) + prefix + 'Cij_l_{}rbins_ext_{}kbins_CAMB_rank{}.bin'.format(red_bin_ext, num_kin, rank) # Cij_l stores Cij for each ell by row
    # Cijl_freader = open(file_Cijl_cross, 'rb') # Open and read a binary file
    # Cijl_sets = np.fromfile(Cijl_freader, dtype='d', count=-1, sep='')
    # #print('Cij(l) from rank', rank, 'is:', Cijl_sets, '\n')
    # Cijl_freader.close()

    #--------------- !! read Gm_cross part by part for each ell -----------------#
    file_Gm_cross = Gm_ifprefix + 'Gm_cross_out_{}rbins_{}kbins_CAMB.bin'.format(red_bin_ext, num_kout)
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
        #offset_cijl = n_l * N_dset * data_type_size
        #offset_Gm = n_l * N_dset * num_kout * data_type_size

        # put the whole array cij at ell into the upper triangle part of the matrix
        cijl_array = Cijl_sets[l*N_dset_ext: (l+1)*N_dset_ext]
        #print(cijl_array, cijl_array.shape)
        cijl_m[iu1] = np.array(cijl_array)
        #print(cijl_m, cijl_m.shape)
        cijl_m_select = np.array(cijl_m[0:red_bin, 0:red_bin])     # select the first red_bin bins of Cij, match the case with T-F
        cijl_true = np.array(cijl_m_select[iu2])                   # convert upper triangle matrix to an array
        cijl_sn = np.array(cijl_true)
        cijl_sn[sn_id] = cijl_true[sn_id] + pseudo_sn              # add shape noise terms in Cii(l) terms

        Cov_cij_cpq = cal_cov_matrix(red_bin, iu2, cijl_sn)        # calculate the covariance matrix of Cij(l), Cpq(l')
        # if rank == 0:
        #     rank_matrix = np.linalg.matrix_rank(Cov_cij_cpq)
        #     print('ell, rank of Cov:', ell, rank_matrix)

        Cov_cij_cpq = Cov_cij_cpq/((2.0*ell+1.0)*delta_l*f_sky)      # account the number of modes for each l with the interval delta_l

        w_ccij, v_ccij = linalg.eigh(Cov_cij_cpq, lower=False, overwrite_a=True)  # Get eigenvalue and eigenvectors from Scipy routine
        w_inv = 1.0/w_ccij
        if not np.all(w_inv>0.0):
            print('w_inv from ell ', ell, ' is negative.')        # show below which ell, the inverse of Cov_cij_cpq fails
        # If uncomment the below, overwrite_a should be set False in the linalg.eigh()
        # sqrt_w_inv = np.diag(w_inv**0.5)
        # v_inv = np.transpose(v_ccij)
        # Cov_cij_sym = np.triu(Cov_cij_cpq, k=1) + Cov_cij_cpq.T
        # print reduce(np.dot, [np.diag(sqrt_w_inv**2.0), v_inv, Cov_cij_sym, v_inv.T])

        Cov_inv_half = np.transpose(w_inv**0.5 *v_ccij)               # Simplify the expression of dot(sqrt_w_inv, v_inv), 05/09/2016
        G_l_array = Gm_sets[l*N_dset_ext*num_kout: (l+1)*N_dset_ext*num_kout]
        Gm_l_ext = np.reshape(G_l_array, (N_dset_ext, num_kout), 'C')         # In Python, the default storage of a matrix follows C language format.
        #print(Gm_l_ext)
        Gm_l = np.array(Gm_l_ext[Gmrow_sel_ind, :])
        Gm_l = np.dot(Cov_inv_half, Gm_l)

        cijl_true = np.dot(Cov_inv_half, cijl_true)

        return cijl_true, Gm_l


    amode = MPI.MODE_WRONLY|MPI.MODE_CREATE

    #-------- generate C^ij(l) prime -------##
    Cijl_prime_file = ofprefix + 'Cijlprime_{}rbins_{}kbins_snf{}_rank{}.bin'.format(red_bin, num_kin, snf, rank)
    Cijl_prime_fwriter = open(Cijl_prime_file, 'wb')

    #------------- !! Gm_prime output
    Gm_prime_file = ofprefix + 'Gm_cross_prime_{}rbins_{}kbins_snf{}_rank{}.bin'.format(red_bin, num_kout, snf, rank)
    Gm_prime_fwriter = open(Gm_prime_file, 'wb')

    Gmrow_sel_ind = np.array([], dtype=int)
    ind_pre = 0
    for row in range(red_bin):
        count = red_bin - row
        for i in range(count):
            Gmrow_sel_ind = np.append(Gmrow_sel_ind, i+ind_pre)
        ind_pre = ind_pre + red_bin_ext - row
    print('Gmrow_sel_ind:', Gmrow_sel_ind)

    iu1 = np.triu_indices(red_bin_ext)
    iu2 = np.triu_indices(red_bin)

    sn_id = [(2*red_bin+1-ii)*ii//2 for ii in range(red_bin)]  # id of dset C^ij(l) which is added with shot noise
    cijl_m = np.zeros((red_bin_ext, red_bin_ext))               # one matrix to store C^ij at one ell
    for l in range(num_l_in_rank):
        cijl_true, Gm_l = cal_C_G(l, rank)
        cijl_true.tofile(Cijl_prime_fwriter, sep="")
        Gm_l.tofile(Gm_prime_fwriter, sep="")

    comm.Barrier()
    Cijl_prime_fwriter.close()
    Gm_prime_fwriter.close()
    t_end = MPI.Wtime()
    if rank == 0:
        print('With total processes', size, ', the running time:', t_end-t_start)

def main():
    cal_sn_dep_Cov_cij()

if __name__ == '__main__':
    main()

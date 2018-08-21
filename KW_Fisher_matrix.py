#!/Users/ding/anaconda3/bin/python
# Copy the code from KW_sn_ratio_Takada_Jain.py, modify it to calculate the Fisher matrix Cijl_wig-Cijl_now, reference is taken from, e.g. Eq. (26) in Hu & Jain 2004.
# We use the covariance matrix of Cijl_wig with conisderation of shape noise. --07/25/2018
# Change the function Fisher_element() to Fisher_matrix() which calculates 2 by 2 Fisher matrix with two parameters alpha and A. --08/06/2018
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
parser = argparse.ArgumentParser(description="Calculate Fisher matrix of Cijl_wig-Cijl_now with MPI implementation, made by Zhejie.", )
parser.add_argument("--nrbin", help = 'Number of tomographic bins.', type=int, required=True)
parser.add_argument("--snf", help = '*The shape noise factor from the default value.', type=float, required=True)
#parser.add_argument("--Pk_type", help = "*The type of input P(k), whether it's linear (Pwig), or damped (Pwig_nonlinear), or without BAO (Pnow).", required=True)
parser.add_argument("--Psm_type", help = '*The expression of Pnorm. The default case, Pnorm from Eisenstein & Zaldarriaga 1999. \
                                          If Pnorm=Pnow, it is derived from transfer function.')
parser.add_argument("--idir0", help = "*The basic directory of input files, e.g., './KW_stage_IV/'.", required=True)
parser.add_argument("--odir0", help = "*The basic directory of output files, e.g., './KW_stage_IV/'.", required=True)
parser.add_argument("--alpha_plus", help = "*The BAO scale shifting parameter alpha larger than 1, i.e. k'=alpha * k or l'= alpha * l given a \chi. Value could be e.g. 1.02.", type=np.float64, required=True)
parser.add_argument("--alpha_minus", help = "*The BAO scale shifting parameter alpha less than 1.", type=np.float64, required=True)
parser.add_argument("--cal_Cijl_wnw", help = "Whether we need to calculate Cijl_wig-Cijl_now for default alpha or shifted alpha.", default='True')

args = parser.parse_args()
num_rbin = args.nrbin
snf = args.snf
#Pk_type = args.Pk_type
Psm_type = args.Psm_type
idir0 = args.idir0
alpha_plus = args.alpha_plus
alpha_minus = args.alpha_minus

prefix = 'Tully-Fisher_'
idir1 = 'mpi_preliminary_data_{}/'
idir_default = idir0 + idir1
##idir_default = idir0 + 'BAO_alpha_1.0/' + idir1
idir_shift_aplus = idir0 + 'BAO_alpha_{}/'.format(alpha_plus) + idir1
idir_shift_aminus = idir0 + 'BAO_alpha_{}/'.format(alpha_minus) + idir1
ifprefix_default = idir_default + prefix                # alpha = 1.0, directory of default data
ifprefix_shift_aplus = idir_shift_aplus + prefix
ifprefix_shift_aminus = idir_shift_aminus + prefix

num_kin = 505
cijl_filename = 'Cij_l_{0}rbins_{1}kbins_CAMB.bin'.format(num_rbin, num_kin)
cijl_wnw_filename = 'Cij_l_wnw_diff_{0}rbins_{1}kbins_CAMB.bin'.format(num_rbin, num_kin)

#------------- calculate the difference between Cijl with and without the BAO wiggles ------
def cal_Cijl_wnw_diff(ifprefix):
    ifile = ifprefix.format('Pwig_nonlinear') + cijl_filename # Cij_l stores Cij for each ell by row
    Cijl_wig = np.fromfile(ifile, dtype='f8', count=-1, sep='')
    ifile = ifprefix.format('Pnow') + cijl_filename
    Cijl_now = np.fromfile(ifile, dtype='f8', count=-1, sep='')
    Cijl_wnw_diff = Cijl_wig - Cijl_now
    ofile = ifprefix.format('Pwig_nonlinear') + 'Cij_l_wnw_diff_{}rbins_{}kbins_CAMB.bin'.format(num_rbin, num_kin)
    Cijl_wnw_diff.tofile(ofile, sep="")

#------------ calculate Fisher matrix elements ---------------------------------------------
def cal_Fisher():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N_dset = (num_rbin+1)*num_rbin//2

    l_min = 1
    #l_max = 31
    l_max = 2002
    delta_l = 3
    num_l = (l_max - l_min)//delta_l + 1
    f_sky = 15000.0/41253.0            # Survey area 15000 deg^2 is from TF-Stage IV
    #print("f_sky: ", f_sky)
    data_type_size = 8

    odir0 = args.odir0
    t_start = MPI.Wtime()

    if Psm_type == 'Pnow':
        Gm_ifprefix = idir0 + 'mpi_preliminary_data_Pwig_nonlinear/set_Pnorm_Pnow/' + prefix
        ofdir = ofdir + 'set_Pnorm_Pnow/'
    else:
        Gm_ifprefix = idir0 + 'mpi_preliminary_data_Pwig_nonlinear/' + prefix

    if rank == 0:
        inputf = Gm_ifprefix + 'pseudo_shapenoise_{0}rbins.out'.format(num_rbin)   # read the default shape noise \sigma^2/n^i
        print(inputf)
        pseudo_sn = np.loadtxt(inputf, dtype='f8', comments='#')
        pseudo_sn = pseudo_sn * snf                                                  # times a shape noise scale factor
        #print(pseudo_sn.shape)
    else:
        pseudo_sn = np.zeros(num_rbin)
    comm.Bcast(pseudo_sn, root=0)

    #------------- !! write output files, they are the basic files --------------#
    ofdir = odir0 + 'BAO_alpha_{}/mpi_{}sn_exp_k_data_Pwig_nonlinear/comm_size{}/'.format(alpha_plus, prefix, size)
    ofprefix = ofdir + prefix
    if rank == 0:
        if not os.path.exists(ofdir):
            os.makedirs(ofdir)

    default_num_l_in_rank = int(np.ceil(num_l / size))
    # Rounding errors here should not be a problem unless default size is very small
    end_num_l_in_rank = num_l - (default_num_l_in_rank * (size - 1))
    assert end_num_l_in_rank >= 1, "Assign fewer number of processes."

    if (rank == (size - 1)):
        num_l_in_rank = end_num_l_in_rank
    else:
        num_l_in_rank = default_num_l_in_rank

    # the length of Cijl array read by each rank
    Cijl_len = num_l_in_rank * N_dset
    def read_Cijl_MPI(Cijl_len, ifile, comm, rank):
        Cijl_target = np.zeros(Cijl_len)
        Cijl_freader = MPI.File.Open(comm, ifile)           # Open and read a binary file
        Cijl_fh_start = rank * Cijl_len * data_type_size    # need to calculate how many bytes shifted
        Cijl_freader.Seek(Cijl_fh_start)
        Cijl_freader.Read([Cijl_target, MPI.DOUBLE])          # Read using individual file pointer
        Cijl_freader.Close()
        return Cijl_target

    # Read the default Cijl_wig from alpha=1.0, with delta_l = 3
    ifile_Cwig_default = ifprefix_default.format('Pwig_nonlinear') + cijl_filename
    Cijl_wig_default = read_Cijl_MPI(Cijl_len, ifile_Cwig_default, comm, rank)

    # Read the default Cijl_now
    ifile_Cwnw_default = ifprefix_default.format('Pwig_nonlinear') + cijl_wnw_filename
    Cijl_wnw_default = read_Cijl_MPI(Cijl_len, ifile_Cwnw_default, comm, rank)

    # Read the Cijl_wnw from shifted ell value with alpha larger than 1.0
    ifile_Cwnw_aplus = ifprefix_shift_aplus.format('Pwig_nonlinear') + cijl_wnw_filename
    Cijl_sets_shift_aplus = read_Cijl_MPI(Cijl_len, ifile_Cwnw_aplus, comm, rank)
    # Read the Cijl_wnw from shifted ell value with alpha smaller than 1.0
    ifile_Cwnw_aminus = ifprefix_shift_aminus.format('Pwig_nonlinear') + cijl_wnw_filename
    Cijl_sets_shift_aminus = read_Cijl_MPI(Cijl_len, ifile_Cwnw_aminus, comm, rank)

    def Fisher_matrix(l, rank):
        n_l = default_num_l_in_rank * rank + l
        ell = l_min + n_l * delta_l
        cijl_wig_default = Cijl_wig_default[l*N_dset: (l+1)*N_dset]
        cijl_wnw_default = Cijl_wnw_default[l*N_dset: (l+1)*N_dset]
        cijl_wig_sn = np.array(cijl_wig_default)                                  # Distinguish the observed C^ijl) (denoted by cijl_wig_sn) from the true C^ij(l)
        cijl_wig_sn[sn_id] = cijl_wig_default[sn_id] + pseudo_sn                  # Add shape noise on C^ii terms to get cijl_wig_sn

        cijl_wnw_shift_aplus = Cijl_sets_shift_aplus[l*N_dset: (l+1)*N_dset]
        cijl_wnw_shift_aminus = Cijl_sets_shift_aminus[l*N_dset: (l+1)*N_dset]
        delta_cijl_wnw = cijl_wnw_shift_aplus - cijl_wnw_shift_aminus
        #print('ell, delta_cijl_wnw:', ell, delta_cijl_wnw)
        Cov_cij_cpq = cal_cov_matrix(num_rbin, iu1, cijl_wig_sn)                  # Get an upper-triangle matrix for Cov(C^ij, C^pq) from Fortran subroutine wrapped.
        Cov_cij_cpq = Cov_cij_cpq.T + np.triu(Cov_cij_cpq, k=1)                   # It's symmetric. Construct the whole matrix for inversion.
        inv_Cov_cij_cpq = linalg.inv(Cov_cij_cpq, overwrite_a=True)
        inv_Cov_cij_cpq = inv_Cov_cij_cpq * ((2.0*ell+1.0)*delta_l*f_sky)             # Take account of the number of modes (the denominator) and f_sky

        dcijl_dalpha = delta_cijl_wnw/(alpha_plus - alpha_minus)
        f_alpalp = reduce(np.dot, [dcijl_dalpha, inv_Cov_cij_cpq, dcijl_dalpha])
        f_alpA = reduce(np.dot, [dcijl_dalpha, inv_Cov_cij_cpq, cijl_wnw_default])     # we need to use cijl_wnw_default to make the model consistent for sigma_alpha from Fisher
        f_AA = reduce(np.dot, [cijl_wig_default, inv_Cov_cij_cpq, cijl_wig_default])
        #f_AA = reduce(np.dot, [cijl_wnw_default, inv_Cov_cij_cpq, cijl_wnw_default])   # It's not correct

        # if rank == 0 and ell == 31:
        #     print('ell, cijl_wnw_default, cijl_wnw_shift, delta_cijl_wnw, inv_Cov_cij_cpq', ell, cijl_wnw_default, cijl_wnw_shift, delta_cijl_wnw, inv_Cov_cij_cpq)
        return ell, f_alpalp, f_alpA, f_AA

    #-------- Output signal-to-noise ratio from each ell -------##
    fisher_ofile = ofprefix + 'fisher_per_ell_{}rbins_{}kbins_snf{}_rank{}_alpha_{}_{}.dat'.format(num_rbin, num_kin, snf, rank, alpha_plus, alpha_minus)
    data_m = np.array([], dtype=np.float64).reshape(0, 4)
    header_line = " ell      f_alpalp    f_alpA    f_AA"
    iu1 = np.triu_indices(num_rbin)
    sn_id = [int((2*num_rbin+1-ii)*ii/2) for ii in range(num_rbin)]          # The ID of dset C^ij(l) added by the shape noise when i=j (auto power components)
    for l in range(num_l_in_rank):
        ell, f_alpalp, f_alpA, f_AA = Fisher_matrix(l, rank)
        data_m = np.vstack((data_m, np.array([ell, f_alpalp, f_alpA, f_AA])))
    print('F_alpalp:', np.sum(data_m[:, 1]), 'from rank:', rank)
    np.savetxt(fisher_ofile, data_m, fmt='%i %.7e %.7e %.7e', delimiter=' ', newline='\n', header=header_line, comments='#')
    sum_fisher = np.zeros(3)
    comm.Barrier()
    comm.Reduce(np.sum(data_m[:, 1:], axis=0), sum_fisher, op=MPI.SUM, root=0)

    t_end = MPI.Wtime()
    if rank == 0:
        Fisher_matrix = np.zeros((2,2))
        Fisher_matrix[0, 0] = sum_fisher[0]
        Fisher_matrix[0, 1] = Fisher_matrix[1, 0] = sum_fisher[1]
        Fisher_matrix[1, 1] = sum_fisher[2]
        print('The Fisher matrix is:', Fisher_matrix)
        inverse_Fisher = linalg.inv(Fisher_matrix, overwrite_a=True)
        print('The inverse Fisher:', inverse_Fisher)#, 'and sigma_alpha=', inverse_Fisher[0, 0]**0.5 * 100))
        sqrt_F_inv = inverse_Fisher**0.5
        print('The (inverse Fisher)^0.5:', sqrt_F_inv)
        print('r_alpA=', inverse_Fisher[0, 1]/(inverse_Fisher[0, 0] * inverse_Fisher[1, 1])**0.5)
        #print('With total processes', size, ', the running time:', t_end-t_start)

def main():
    if args.cal_Cijl_wnw == 'True':
        cal_Cijl_wnw_diff(ifprefix_default)
        cal_Cijl_wnw_diff(ifprefix_shift_aplus)
        cal_Cijl_wnw_diff(ifprefix_shift_aminus)
    cal_Fisher()

if __name__ == '__main__':
    main()

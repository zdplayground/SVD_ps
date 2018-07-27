#!/Users/ding/anaconda3/bin/python
# Copy the code from KW_Fisher_matrix.py, modify it to calculate the Fisher matrix Cijl_wig-Cijl_now for PW stage surveys. --07/27/2018
#
from mpi4py import MPI
import numpy as np
import os, sys
from scipy import integrate
from scipy import linalg
from functools import reduce
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/TW_f2py_SVD/')
from cov_matrix_module import cal_cov_matrix
import argparse

#---------------------------------------------------------
parser = argparse.ArgumentParser(description="Calculate Fisher matrix of Cijl_wig-Cijl_now, made by Zhejie.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--nrbin_case", help = 'The case of number of tomographic bins.', type=int, required=True)
parser.add_argument("--snf", help = '*The shape noise factor from the default value.', type=float, required=True)
#parser.add_argument("--Pk_type", help = "*The type of input P(k), whether it's linear (Pwig), or damped (Pwig_nonlinear), or without BAO (Pnow).", required=True)
parser.add_argument("--Psm_type", help = '*The expression of Pnorm. The default case, Pnorm from Eisenstein & Zaldarriaga 1999. \
                                          If Pnorm=Pnow, it is derived from transfer function.')
parser.add_argument("--idir0", help = "*The basic directory of input files, e.g., './PW_stage_III/'.", required=True)
parser.add_argument("--odir0", help = "*The basic directory of output files, e.g., './PW_stage_III/'.", required=True)
parser.add_argument("--alpha", help = "The BAO scale shifting parameter alpha, i.e. k'=alpha * k or l'= alpha * l given a \chi. Value could be e.g. 1.02.", required=True, type=np.float64)

args = parser.parse_args()
nrbin_case = args.nrbin_case
snf = args.snf
#Pk_type = args.Pk_type
Psm_type = args.Psm_type
idir0 = args.idir0
alpha = args.alpha

prefix = 'TW_zext_'
num_rbin = {'num_sbin': (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 22, 25, 27, 30, 32, 35, 37),
            'num_pbin': (2, 3, 5, 6, 7, 8, 10, 11, 12, 18, 25, 27, 31, 34, 37, 40, 44, 46)}    # s: spectroscopic; p: photometric
red_bin = num_rbin['num_sbin'][nrbin_case]
red_bin_ext = num_rbin['num_pbin'][nrbin_case]

idir1 = 'mpi_preliminary_data_{}/'
idir_default = idir0 + idir1
##idir_default = idir0 + 'BAO_alpha_1.0/' + idir1
idir_shift = idir0 + 'BAO_alpha_{}/'.format(alpha) + idir1
ifprefix_default = idir_default + prefix
ifprefix_shift = idir_shift + prefix

num_kin = 505
cijl_filename = 'Cij_l_{0}rbins_ext_{1}kbins_CAMB.bin'.format(red_bin_ext, num_kin)

#------------- calculate the difference between Cijl with and without the BAO wiggles ------
def cal_Cijl_wnw_diff(ifprefix):
    ifile = ifprefix.format('Pwig_nonlinear') + cijl_filename # Cij_l stores Cij for each ell by row
    Cijl_wig = np.fromfile(ifile, dtype='f8', count=-1, sep='')
    ifile = ifprefix.format('Pnow') + cijl_filename
    Cijl_now = np.fromfile(ifile, dtype='f8', count=-1, sep='')
    Cijl_wnw_diff = Cijl_wig - Cijl_now
    ofile = ifprefix.format('Pwig_nonlinear') + 'Cij_l_wnw_diff_{}rbins_{}kbins_CAMB.bin'.format(red_bin, num_kin)
    Cijl_wnw_diff.tofile(ofile, sep="")

#------------ calculate Fisher matrix elements ---------------------------------------------
def cal_Fisher():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N_dset = (red_bin+1) * red_bin//2
    N_dset_ext = (red_bin_ext+1) * red_bin_ext//2
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
        inputf = Gm_ifprefix + 'pseudo_shapenoise_{0}rbins_ext.out'.format(red_bin_ext)     # read the default shape noise \sigma^2/n^i
        print(inputf)
        pseudo_sn = np.loadtxt(inputf, dtype='f8', comments='#')
        pseudo_sn = pseudo_sn[0: red_bin] * snf                                                  # times a shape noise scale factor
        #print(pseudo_sn.shape)
    else:
        pseudo_sn = np.zeros(red_bin)
    comm.Bcast(pseudo_sn, root=0)

    #------------- !! write output files, they are the basic files --------------#
    ofdir = odir0 + 'BAO_alpha_{}/mpi_Fisher_matrix_element/comm_size{}/'.format(alpha, prefix, size)
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

    # be careful here we have extended photometric redshift bins, which is different from TF case.
    Cijl_len = num_l_in_rank * N_dset_ext
    Cijl_wig_default = np.zeros(Cijl_len)
    Cijl_sets_default = np.zeros(Cijl_len)
    Cijl_sets_shift = np.zeros(Cijl_len)

    cijl_wnw_filename = 'Cij_l_wnw_diff_{0}rbins_{1}kbins_CAMB.bin'.format(red_bin, num_kin)
    # Read the default Cijl, with delta_l = 3
    file_Cijl_cross = ifprefix_default.format('Pwig_nonlinear') + cijl_wnw_filename
    Cijl_freader = MPI.File.Open(comm, file_Cijl_cross) # Open and read a binary file
    Cijl_fh_start = rank * Cijl_len * data_type_size    # need to calculate how many bytes shifted
    Cijl_freader.Seek(Cijl_fh_start)
    Cijl_freader.Read([Cijl_sets_default, MPI.DOUBLE])          # Read using individual file pointer
    #print('Cij(l) from rank', rank, 'is:', Cijl_sets, '\n')
    comm.Barrier()
    Cijl_freader.Close()

    file_Cijl_cross = ifprefix_default.format('Pwig_nonlinear') + cijl_filename
    Cijl_freader = MPI.File.Open(comm, file_Cijl_cross) # Open and read a binary file
    Cijl_fh_start = rank * Cijl_len * data_type_size    # need to calculate how many bytes shifted
    Cijl_freader.Seek(Cijl_fh_start)
    Cijl_freader.Read([Cijl_wig_default, MPI.DOUBLE])          # Read using individual file pointer
    comm.Barrier()
    Cijl_freader.Close()

    # Read the Cijl with shift ell value
    file_Cijl_cross = ifprefix_shift.format('Pwig_nonlinear') + cijl_wnw_filename
    Cijl_freader = MPI.File.Open(comm, file_Cijl_cross) # Open and read a binary file
    Cijl_fh_start = rank * Cijl_len * data_type_size    # need to calculate how many bytes shifted
    Cijl_freader.Seek(Cijl_fh_start)
    Cijl_freader.Read([Cijl_sets_shift, MPI.DOUBLE])          # Read using individual file pointer
    comm.Barrier()
    Cijl_freader.Close()

    def select_elements(data_ext, iu1, iu2):
        l1, l2 = red_bin_ext, red_bin
        m1 = np.zeros((l1, l1), dtype=np.float64)
        m1[iu1] = data_ext
        m2 = m1[0: l2, 0: l2]
        data_obj = m2[iu2]
        return data_obj

    def Fisher_element(l, rank):
        n_l = default_num_l_in_rank * rank + l
        ell = l_min + n_l * delta_l
        cijl_default = Cijl_sets_default[l*N_dset_ext: (l+1)*N_dset_ext]
        cijl_wig_default = Cijl_wig_default[l*N_dset_ext: (l+1)*N_dset_ext]

        cijl_sn = select_elements(cijl_wig_default, iu1, iu2)
        cijl_sn[sn_id] = cijl_wig_default[sn_id] + pseudo_sn                  # Add shape noise on C^ii terms to get cijl_sn

        cijl_shift = Cijl_sets_shift[l*N_dset_ext: (l+1)*N_dset_ext]
        cijl_diff = cijl_shift - cijl_default
        cijl_diff = select_elements(cijl_diff, iu1, iu2)

        #print('ell, cijl_diff:', ell, cijl_diff)
        Cov_cij_cpq = cal_cov_matrix(red_bin, iu2, cijl_sn)               # Get an upper-triangle matrix for Cov(C^ij, C^pq) from Fortran subroutine wrapped.
        Cov_cij_cpq = Cov_cij_cpq.T + np.triu(Cov_cij_cpq, k=1)           # It's symmetric. Construct the whole matrix for inversion.
        inv_Cov_cij_cpq = linalg.inv(Cov_cij_cpq, overwrite_a=True)
        inv_Cov_cij_cpq = inv_Cov_cij_cpq * ((2.0*ell+1.0)*delta_l*f_sky)             # Take account of the number of modes (the denominator) and f_sky
        #print('ell, inv_Cov_cij_cpq', ell, inv_Cov_cij_cpq)
        #dcijl_dalpha = cijl_diff/(alpha-1.0)
        dcijl_dalpha = cijl_diff/(alpha-1.0)
        faa = reduce(np.dot, [dcijl_dalpha, inv_Cov_cij_cpq, dcijl_dalpha])

        # if rank == 0 and ell == 31:
        #     print('ell, cijl_default, cijl_shift, cijl_diff, inv_Cov_cij_cpq', ell, cijl_default, cijl_shift, cijl_diff, inv_Cov_cij_cpq)
        return ell, faa

    #-------- Output signal-to-noise ratio from each ell -------##
    faa_ofile = ofprefix + 'faa_per_ell_{}rbins_{}kbins_snf{}_rank{}.dat'.format(num_rbin, num_kin, snf, rank)
    data_m = np.array([], dtype=np.float64).reshape(0, 2)
    header_line = " ell      faa"

    iu1 = np.triu_indices(red_bin_ext)
    iu2 = np.triu_indices(red_bin)
    sn_id = [int((2*red_bin+1-ii)*ii/2) for ii in range(red_bin)]          # The ID of dset C^ij(l) added by the shape noise when i=j (auto power components)
    Faa = 0.0
    for l in range(num_l_in_rank):
        ell, faa = Fisher_element(l, rank)
        data_m = np.vstack((data_m, np.array([ell, faa])))
        Faa = Faa + faa
    print('Faa:', Faa, 'from rank:', rank)
    np.savetxt(faa_ofile, data_m, fmt='%.7e', delimiter=' ', newline='\n', header=header_line, comments='#')

    Faa_total = comm.reduce(Faa, op=MPI.SUM, root=0)
    comm.Barrier()
    t_end = MPI.Wtime()
    if rank == 0:
        print('The total Fisher Faa is:', Faa_total, 'sigma alpha (%)=', 1./Faa_total**0.5 * 100)
        print('With total processes', size, ', the running time:', t_end-t_start)

def main():
    #cal_Cijl_wnw_diff(ifprefix_default)
    #cal_Cijl_wnw_diff(ifprefix_shift)
    cal_Fisher()

if __name__ == '__main__':
    main()

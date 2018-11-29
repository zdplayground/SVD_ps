#!/Users/ding/miniconda3/bin/python
# 1. Copied the code TF_svd_cov_p_cross_bin.py from the folder ../KW_stage_IV/, modify it for data files in pseudo_PW_stage_IV. -- 10/19/2017
# Make it appliable for files in both pseudo_PW_stage_IV and pseudo_KW_stage_IV. -- 10/30/2017
# 2. Similar as code svd_CovPwig_over_Pnow_nSV.py, we add the option whether we set SVc on the output inverse covariance matrix of Pwnw. The difference
# from svd_CovPwig_over_Pnow_nSV.py is that we use P_{now, input} as the scale factor transfering inverse covariance matrix between Pwig and Pwnw. --05/04/2018
# 3. Add the option "set_SVc_on_Pk" to include the case that we apply SVc on both Pk and CovP. -- 05/16/2018
# 4. Add the argument "start_lmin" to skip small angular modes of C^ij(l) and G matrix due to f_sky effect. --06/05/2018
# 5. Output the covariance matrix of Pwig. --11/16/2018
#--------------------------------------------------------------------------------------------------------#
# This code manages to play with SVD to find a good way to invert Cov^-1(P).
# What's the ruler to measure the goodness of inversion? (See singular value decomposition method in Numerical Recipe or online.)
# The important thing is that we should use SVD to decompose G', then use the solution of SVD(G') to express Cov_P,
# instead of SVD Cov_P directly.
# It's the same as svd_cov_p1.py, it only deals with shape noise included in the covariance matrix of C^ij(l)!!
# Modify details in the plotting to give figures in the writing of prospectus. Output figures in figs_prospectus folder. --03/11/2017.
# Polish the code using Python3 version. Orgainze the output files. -- 08/14/2017
# (Need to figure out how to normalize the covariance matrix of output P(k). -- 08/29/2017)--no need to do that. 09/26/2017
# ----------------------------------
# Modify it to read Cijl and Gm from multiple binary data files. -- 10/24/2017
#
import os, sys
import time
import numpy as np
import scipy
from functools import reduce  # for Python3
from scipy import interpolate
import matplotlib.pyplot as plt
import cosmic_params
import argparse
import pylab

parser = argparse.ArgumentParser(description='Show the extracted P(k) from SVD, made by Zhejie.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lt", help = '*Weak lensing survey type, TF for Tully-Fisher case; TW for traditional case.', required=True)
parser.add_argument("--nrbin", help = '*Number of tomographic bins.', type=int, required=True)
parser.add_argument("--nkout", help = '*Number of output k bins.', type=int, required=True)
parser.add_argument("--start_lmin", help = "*The minimum ell value considered in the analysis. Default is 1. For Stage III, it's 10 for Stage III and 4 for Stage IV", default=1, type=int)
parser.add_argument("--shapenf", help = "*Shape noise factor.", type=float, required=True)
parser.add_argument("--Pk_type", help = '*Type of P(k). Pwig_linear: containing BAO; Pnow: without BAO; Pwig_nonlinear: with damped BAO.', required=True)
parser.add_argument("--nrank", help = "*The total number of processes used in MPI to generate Cijl_prime", type=int, required=True)
parser.add_argument("--Psm_type", help = 'The expression of Pnorm. The default case, Pnorm from Eisenstein & Zaldarriaga 1999. \
                                          Test Pnorm=Pnow, which is derived from transfer function.')
parser.add_argument("--output_Pk_type", help = '*Weather the output is Pwig or Pwig/Pnow where Pnow is the spline fit from transfer function. \
                    Type Pk_now, Pk_wig or Pk_wnw. If Pk_now, make sure the input Pk_type=Pnow.', required=True)
parser.add_argument("--idir0", help = "*The basic input file directory, e.g., './PW_stage_III/'.", required=True)
parser.add_argument("--odir0", help = "*The basic output file directory, e.g., './PW_stage_III/'.", required=True)
parser.add_argument("--scale_nz", help = 'The scale factor of total angular galaxy number density. For KW-stage IV, the default is 1.1 (no need to give),\
                    and the optimistic case is 4.3/arcmin^2', type=float)
parser.add_argument("--set_SVc_on_Pk", help = "*Whether we replace smaller SV to be SVc in the W matrix for the ouput Pk. Either True or False. If it's True, the set_SVc_on_CovP should be True too.", required=True)
parser.add_argument("--set_SVc_on_CovP", help = '*Whether we replace smaller SV to be SVc in W matrix for the output inverse covariance matrix of Pk. Either True or False.', required=True)
parser.add_argument("--modify_Cov_cij_cpq", help = "Whether we have modified the Cov_cij_cpq to output Cijl^prime and Gm^prime or not. Only for PW surveys.", required=True)

args = parser.parse_args()
lt = args.lt
num_rbin = args.nrbin                   # nrbin represents the number of tomographic bins
num_kout = args.nkout                   # number of k output bins for output power spectrum
start_lmin = args.start_lmin
shapen_factor = args.shapenf
Pk_type = args.Pk_type                  # Simulate lensing power spectrum from 3D power spectrum P(k) with/without BAO.
nrank = args.nrank
Psm_type = args.Psm_type                # the expression of Pnorm, either the default case or Pnow from transfer function.
output_Pk_type = args.output_Pk_type                # Define output P(k) type.
idir0 = args.idir0
odir0 = args.odir0
scale_nz = args.scale_nz
set_SVc_on_Pk = args.set_SVc_on_Pk
set_SVc_on_CovP = args.set_SVc_on_CovP
modify_Cov_cij_cpq = args.modify_Cov_cij_cpq

# input an eigenvalue array, return an inversed eigenvalue matrix for the output inverse covariance matrix
def fun_eigenv_m_Cov(A, boundary):
    A_size = np.size(A)
    A_inv = np.zeros((A_size, A_size))
    A_matrix = np.zeros((A_size, A_size))
    for jj in range(A_size):
        #if abs(A[jj]) > boundary:
        if A[jj] >= boundary:
            A_inv[jj][jj] = 1.0/A[jj]
            A_matrix[jj][jj] = A[jj]
        else:
            if set_SVc_on_CovP == 'True':
                A_inv[jj][jj] = 1.0/boundary
                A_matrix[jj][jj] = boundary
            else:
                break
    return A_matrix, A_inv

# similar as fun_eigenv_m_Cov, but for output P(k). We ignore modes with SV smaller than SVc.
def fun_eigenv_m_Pk(A, boundary):
    A_size = np.size(A)
    A_inv = np.zeros((A_size, A_size))
    A_matrix = np.zeros((A_size, A_size))
    for jj in range(A_size):
        #if abs(A[jj]) > boundary:
        if A[jj] >= boundary:
            A_inv[jj][jj] = 1.0/A[jj]
            A_matrix[jj][jj] = A[jj]
        else:
            if set_SVc_on_Pk == 'True':
                A_inv[jj][jj] = 1.0/boundary
                A_matrix[jj][jj] = boundary
            else:
                break
    return A_matrix, A_inv

def extract_Cov_Pk(V_T, W, W_b, Uinv, Cijlprime, Pnorm_out):
    V = np.transpose(V_T)
    W_m_Cov, W_m_inv_Cov = fun_eigenv_m_Cov(W, W_b) # set boundary point for W, it depends
    #print W_m_inv
    Cov_P_inv = reduce(np.dot, [V, W_m_Cov**2.0, V_T])
    Cov_P = reduce(np.dot, [V, W_m_inv_Cov**2.0, V_T])
    #print Cov_P
    W_m_Pk, W_m_inv_Pk = fun_eigenv_m_Pk(W, W_b)
    Pk_prime = reduce(np.dot, [V, W_m_inv_Pk, Uinv, Cijlprime])
    Pk_prime = Pk_prime*Pnorm_out
    # It seems I didn't normalize Cov_P here.
    return Cov_P_inv, Cov_P, Pk_prime


def plot_Pk(k_mid, Pk_prime, sigma_P1, kcamb, Pkcamb_prime, W, num_rbin, num_kout, shapen_factor, num_eigv_W, lt_prefix, odir):
    plt.figure(figsize=(12, 8.5))
    #show Pk from SVD and error bars
    plt.errorbar(k_mid, Pk_prime, yerr=sigma_P1, elinewidth=2.0, capsize=3.0, linewidth=2.0, label='SVD')#check 1 sigma
    #plt.plot(kout, Pk_prime, label='From SVD')
    plt.plot(kcamb, Pkcamb_prime, 'r', linewidth=2.0, label='CAMB')
    plt.xscale("log")
    plt.xlim([1.e-2, 2.5])
    plt.xticks(fontsize=24)
    plt.ylim([0., 2.0])
    plt.yticks([0.0, 0.5, 1.0, 1.5], fontsize=24)
    textline = '{0} SV\n'.format(num_eigv_W)+'max={0:.3e}\n'.format(W[0])+'min={0:.3e}'.format(W[num_eigv_W-1])
    ##textloc = [[0.2, 0.5], [0.2, 2.0]]
    textloc = [0.2, 0.5]
    plt.text(textloc[0], textloc[1], textline, fontsize=24)
    plt.grid(True)
    plt.legend(loc='upper right', frameon=False, fontsize=24)
    plt.xlabel(r'$k$'+' '+r'$[h Mpc^{-1}]$', fontsize=24)
    #plt.ylabel(r'$P_{\delta}(k)$'+' '+'$(Mpc^3/h^3)$')
    plt.ylabel(r'$P_{\delta}(k)/P_s(k)$', fontsize=24)
    #plt.suptitle('{0} SV'.format(num_eigv_W), fontsize=14, fontweight='bold')
    figname = odir + '{}_svd_P_{}rbins_{}kbins_snf{}_{}eigenvW.pdf'.format(lt_prefix[lt], num_rbin, num_kout, shapen_factor, num_eigv_W)
    plt.savefig(figname)
    ##plt.show()
    plt.close()


##-----------------------------------------------------------------------------------------##
##----------------------- Show extracted P(k) from shear power spectrum -------------------##
##-----------------------------------------------------------------------------------------##
# Try using different numbers of eigenvalues from SVD method
def extract_Pk_svd():
    N_dset = (num_rbin+1)*num_rbin//2

    num_inputk = 505
    l_min = 1
    l_max = 2002
    #delta_l = 1
    delta_l = 3
    num_l = int((l_max-l_min)/delta_l + 1)
    lt_prefix = {'TF': 'Tully-Fisher', 'TW': 'TW_zext'}

    idir1 = '/mpi_{}_sn_exp_k_data_{}/comm_size{}/'.format(lt_prefix[lt], Pk_type, nrank)
    if modify_Cov_cij_cpq == 'True':
        idir1 = idir1 + 'modify_Cov_cij_cpq/'

    idir = idir0 + idir1

    if Psm_type == 'Pnow':
        idir = idir + 'set_Pnorm_Pnow/'

    inputf = '/Users/ding/Documents/playground/shear_ps/SVD_ps/Input_files/CAMB_Planck2015_matterpower.dat'
    kcamb, Pkcamb = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
    tck_Pk = interpolate.splrep(kcamb, Pkcamb)
    k_0 = 0.001       # unit h*Mpc^-1
    Pk_0 = interpolate.splev(k_0, tck_Pk, der=0)
    ## add normalization term P_0 = A*k^n*T^2, where T is the transfer function, set n=1, P_CAMB=P_0 at k=0.001hMpc^-1
    # kk has the same value as k

    inputf = '/Users/ding/Documents/playground/shear_ps/SVD_ps/Input_files/transfer_fun_Planck2015.dat'
    kk, Tf = np.loadtxt(inputf, dtype='f8', comments='#', usecols=(0,1), unpack=True)
    #print kk==kcamb
    tck_Tf = interpolate.splrep(kk, Tf)
    Tf_0 = interpolate.splev(k_0, tck_Tf, der=0)

    P0_a = Pk_0/(pow(k_0, cosmic_params.ns) *Tf_0**2.0)
    P0 = P0_a * pow(kcamb, cosmic_params.ns) * Tf**2.0        # get primordial power spectrum from the transfer function
    tck_Pnw = interpolate.splrep(kcamb, P0)

    Pkcamb_prime = Pkcamb/P0                    # emphasize the oscillation shape of BAO

    if lt == 'TF':
        odir1 = '/{}_Pk_output_dset_{}/'.format(lt, Pk_type)
    else:
        odir1 = '/{}_Pk_output_dset_{}/'.format(lt_prefix[lt], Pk_type)

    odir2 = '{}rbins_{}kbins_snf{}/'.format(num_rbin, num_kout, shapen_factor)
    if scale_nz:
        odir2 = '{}rbins_{}kbins_snf{}_nz{}/'.format(num_rbin, num_kout, shapen_factor, scale_nz)

    if Psm_type == 'Pnow':
        odir2 = odir2 + 'set_Pnorm_Pnow/'.format(num_rbin, num_kout, shapen_factor)

    if start_lmin != 1:
        odir2 = odir2 + 'start_ell_{}/'.format(start_lmin)
    if modify_Cov_cij_cpq == 'True':
        odir2 = odir2 + 'modify_Cov_cij_cpq/'

    odir0 = args.odir0
    if set_SVc_on_CovP == 'True' and set_SVc_on_Pk != 'True':
        odir0 = odir0 + 'apply_SVc_on_CovP/'
    elif set_SVc_on_CovP == 'True' and set_SVc_on_Pk == 'True':
        odir0 = odir0 + 'apply_SVc_on_Pk_and_CovP/'

    odir_Pk = odir0 + odir1 + odir2
    if not os.path.exists(odir_Pk):
        os.makedirs(odir_Pk)

    if Pk_type != 'Pnow':
        odir_figs = odir0 + '/{}_Pk_output_figs_{}/'.format(lt_prefix[lt], Pk_type) + odir2
        if not os.path.exists(odir_figs):
            os.makedirs(odir_figs)

    # ofile = odir_Pk + 'Pk_woversm_camb.dat'
    # header_line = 'k_camb       Pk_woversm_camb'
    # np.savetxt(ofile, np.array([kcamb, Pkcamb_prime]).T, fmt='%.7e', header = header_line, comments='#', newline='\n')

    ##eigv_a = [i+count*i/4 for count in range(10)]
    eigv_a = np.arange(1, num_kout+1)

    def normalize_cov(Cov_P, num_eigv_W):
        for i in range(np.size(Cov_P, axis=0)):
            for j in range(np.size(Cov_P, axis=1)):
                Cov_P[i, j] = Cov_P[i, j]/(Cov_P[i, i]*Cov_P[j, j])
        return Cov_P

    def save_Pk():
        #--------------------------------------set Pnorm_out-------------------------------------#
        kout, k_mid, Pnorm_out = np.zeros(num_kout+1), np.zeros(num_kout), np.zeros(num_kout)
        k_low, k_high = 0.01, 1.0
        kout[0], kout[1], kout[-1] = kcamb[0], k_low, kcamb[-1]
        k_factor = pow(k_high/k_low, 1.0/(num_kout-2))

        for i in range(2, num_kout):
            kout[i] = kout[i-1]*k_factor
        #print kout
        for i in range(num_kout):
            k_mid[i] = (kout[i] + kout[i+1])/2.0

        if Psm_type == 'Pnow':
            Pnorm_out[:] = 1.0
        else:
            # add transfer function term on Pnorm_out, it's like a new normalization
            if output_Pk_type == 'Pk_wnw':
                Pk_scale_factor = interpolate.splev(k_mid, tck_Pnw, der=0)
            elif output_Pk_type == 'Pk_wig' or output_Pk_type == 'Pk_now':
                Pk_scale_factor = 1.0
            Pnorm_out = 1.5e4/(1.0+(k_mid/0.05)**2.0)**0.65/Pk_scale_factor

        #-----------------------------------------------------------------------------------------#
        # default case with delta_l =3
        Cijlprime = np.array([], dtype=np.float64)
        Gm_cross_prime = np.empty((0, num_kout), dtype=np.float64)
        default_num_l_in_rank = int(np.ceil(num_l/nrank))
        end_num_l_in_rank = num_l - default_num_l_in_rank * (nrank-1)
        print('default # l, end # l:', default_num_l_in_rank, end_num_l_in_rank)
        for rank in range(nrank):
            filename = '{}_Cijlprime_{}rbins_{}kbins_snf{}_rank{}.bin'.format(lt_prefix[lt], num_rbin, num_inputk, shapen_factor, rank)
            if scale_nz:
                filename = '{}_Cijlprime_{}rbins_{}kbins_snf{}_nz{}_rank{}.bin'.format(lt_prefix[lt], num_rbin, num_inputk, shapen_factor, scale_nz, rank)
            file_Cijlprime = idir + filename
            print(file_Cijlprime)
            Cijlprime_freader = open(file_Cijlprime, 'rb')
            if start_lmin != 1 and rank == 0:
                len_dump = (start_lmin-l_min)//delta_l * N_dset
                dump_data = np.fromfile(Cijlprime_freader, dtype='d', count=len_dump, sep='')  # skip the data of low ell mode
            Cijlprime_part = np.fromfile(Cijlprime_freader, dtype='d', count=-1, sep='')

            Cijlprime_freader.close()
            print("Cijlprime_part:", len(Cijlprime_part))
            Cijlprime = np.append(Cijlprime, Cijlprime_part)

            # read file from Gprime_out which gives output P'(k) from Ciil
            filename = '{}_Gm_cross_prime_{}rbins_{}kbins_snf{}_rank{}.bin'.format(lt_prefix[lt], num_rbin, num_kout, shapen_factor, rank)
            if scale_nz:
                filename = '{}_Gm_cross_prime_{}rbins_{}kbins_snf{}_nz{}_rank{}.bin'.format(lt_prefix[lt], num_rbin, num_kout, shapen_factor, scale_nz, rank)
            file_Gm_cross_prime = idir + filename
            print(file_Gm_cross_prime)
            Gm_cross_prime_freader = open(file_Gm_cross_prime, 'rb')
            l_start = 0
            if rank == nrank - 1:
                num_l_in_rank = end_num_l_in_rank
            else:
                num_l_in_rank = default_num_l_in_rank

            if start_lmin != 1 and rank == 0:
                l_start = (start_lmin - l_min)//delta_l
                dump_data = np.fromfile(Gm_cross_prime_freader, dtype='d', count = N_dset*num_kout*l_start, sep='')

            for l in range(l_start, num_l_in_rank):
                 G_l_array = np.fromfile(Gm_cross_prime_freader, dtype='d', count=N_dset*num_kout, sep='')
                 Gm_cp_l = np.reshape(G_l_array, (N_dset, num_kout))
                 Gm_cross_prime = np.append(Gm_cross_prime, Gm_cp_l, axis=0)
            Gm_cross_prime_freader.close()

        print('Gm_cross_prime', Gm_cross_prime)
        Gm_cross_prime_T = np.transpose(Gm_cross_prime)
        Cov_P_prime_inv = np.dot(Gm_cross_prime_T, Gm_cross_prime)
        Cov_P_inv = np.zeros(Cov_P_prime_inv.shape)
        for i in range(num_kout):
            for j in range(num_kout):
                Cov_P_inv[i, j] = Cov_P_prime_inv[i, j]/(Pnorm_out[i] * Pnorm_out[j])

        if output_Pk_type == 'Pk_wnw':
            ofile = odir_Pk + 'Cov_Pwnw_inv_{}rbin_{}kbin_withshapenoisefactor{}.npz'.format(num_rbin, num_kout, shapen_factor)
            print('Cov_P_inv: ', Cov_P_inv)
            np.savez(ofile, Cov_P_inv)
            print("Cov_P_inv: ", Cov_P_inv)
        # pylab.pcolor(Cov_P_inv)
        # pylab.colorbar()
        # pylab.show()
        #print np.amax(Cov_P_inv), np.amin(Cov_P_inv)

        U, W, V_T = scipy.linalg.svd(Gm_cross_prime, full_matrices = False, overwrite_a=True)
        Uinv = np.transpose(U)

        # calculate U_j^T Cijlprime, w character denotes Cijlprime
        Ujw = np.array([np.dot(U[:, i], Cijlprime) for i in range(np.size(U, axis=1))])
        print("Ujw': ", Ujw)
        # calculate Wj^-1 Uj^T Cijlprime
        WinvUw = Ujw/W
        print("W^{-1} Uj W': ", WinvUw)

        for i in range(np.size(Ujw)):
            if abs(Ujw[i])<1.0:
                print('Ujw_boundary_id:', i)
                break
        print(W[i-1], W[i])
        if output_Pk_type == 'Pk_wnw':
            # save eigenvalues and information as Table 2 in Eisenstein & Zaldarriaga 1999.
            ofile = odir_Pk + 'eigenvalues_{}rbin_{}kbin_withshapenoisefactor{}.out'.format(num_rbin, num_kout, shapen_factor)
            header_line = " The eigenvalues from SVD with descending order, as well as the derived information\n"
            header_line += " Wj      U_j^T w'     W_j^{-1} U_j^T w' "
            np.savetxt(ofile, np.array([W, Ujw, WinvUw]).T, fmt='%.7e', header=header_line, comments='#', newline='\n')


        # It's may not be a good judgment to get the boundary of Wj
        for j in range(np.size(WinvUw)):
            if abs(WinvUw[j]) > 10.0:
                print('WinUw_boundary_id:', j)
                break
        print("The last two eigenvalues: ", W[j-1], W[j])

        # set boundary W_b for eigenvalue array
        i += 1
        j += 1

        for num_eigv_W in eigv_a:
            if num_eigv_W <= num_kout:
                W_b = W[num_eigv_W -1]
                Cov_P_prime_inv, Cov_P_prime, Pk_prime = extract_Cov_Pk(V_T, W, W_b, Uinv, Cijlprime, Pnorm_out)
                #print('Cov_P_prime: ', Cov_P_prime)

                sigma_P1 = np.diag(Cov_P_prime)**0.5
                sigma_P1 = sigma_P1 * Pnorm_out # ! it's the normalization
                datam = np.array([k_mid, Pk_prime, sigma_P1]).T

                ofile = odir_Pk + '{}_{}rbin_{}kbin_withshapenoisefactor{}_{}eigenvW.out'.format(output_Pk_type, num_rbin, num_kout, shapen_factor, num_eigv_W)
                header_line = ' k_out   {}_out    sigma_P'.format(output_Pk_type)
                np.savetxt(ofile, datam, fmt='%.7e', header=header_line, newline='\n', comments='#')

                if set_SVc_on_CovP == 'True':
                    Cov_Pwnw_inv = np.zeros(Cov_P_prime_inv.shape)
                    for i in range(num_kout):
                        for j in range(num_kout):
                            Cov_Pwnw_inv[i, j] = Cov_P_prime_inv[i, j]/(Pnorm_out[i] * Pnorm_out[j])
                    if output_Pk_type == 'Pk_wnw':
                        filename = 'Cov_Pwnw_inv_{}rbin_{}kbin_withshapenoisefactor{}_{}eigenvW_SVc.npz'.format(num_rbin, num_kout, shapen_factor, num_eigv_W)
                    else:
                        filename = 'Cov_{}_inv_{}rbin_{}kbin_withshapenoisefactor{}_{}eigenvW_SVc.npz'.format(output_Pk_type, num_rbin, num_kout, shapen_factor, num_eigv_W)
                    ofile = odir_Pk + filename
                    np.savez(ofile, Cov_Pwnw_inv)

    save_Pk()

    def show_Pk(Pkcamb_prime):
        for num_eigv_W in eigv_a:
            ifile = odir_Pk + 'Pk_wnw_{}rbin_{}kbin_withshapenoisefactor{}_{}eigenvW.out'.format(num_rbin, num_kout, shapen_factor, num_eigv_W)
            k_mid, Pk_prime, sigma_P1 = np.loadtxt(ifile, dtype='f8', comments='#', unpack=True)
            ifile = odir_Pk + 'eigenvalues_{}rbin_{}kbin_withshapenoisefactor{}.out'.format(num_rbin, num_kout, shapen_factor)
            W = np.loadtxt(ifile, dtype='f8', comments='#', usecols=(0, ))

            if Pk_type == 'Pnow':
                Pkcamb_prime = np.ones(len(kcamb))

            plot_Pk(k_mid, Pk_prime, sigma_P1, kcamb, Pkcamb_prime, W, num_rbin, num_kout, shapen_factor, num_eigv_W, lt_prefix, odir_figs)

    if output_Pk_type == 'Pk_wnw':
        show_Pk(Pkcamb_prime)


def main():
    t0 = time.time()
    extract_Pk_svd()
    t1 = time.time()
    print("Running time (s): ", t1-t0)
    #show_rij_Cov()

if __name__ == '__main__':
    main()

#!/Users/ding/anaconda3/bin/python
#  Copy the code from ../KW_stage_IV, modify it for stage_III setting.    --09/19/2018
# ------------------------------------------------------------------------------------------------------------------------------
# 1. scale_n has been applied on n(z) to match the total angular density.
# 2. Since chi_array is unitless from data file, tranform chi_array (times it by c/H_0) to be unit of Mpc/h
# 3. Change the name of basic output directory to be BAO_alpha_xx_v2 (version 2).
# 4. Set the difference between chibin[i+1] and chi_k larger than 1.e-8 to avoid possible numerical error on g_i. --09/16/2018
#
from mpi4py import MPI
import numpy as np
import math
import os, sys
sys.path.append("/Users/ding/Documents/playground/shear_ps/SVD_ps/")
import cosmic_params
sys.path.append("/Users/ding/Documents/playground/shear_ps/SVD_ps/KW_modules/")
sys.path.append("/Users/ding/Documents/playground/shear_ps/SVD_ps/common_modules/")
from lens_eff_module import lens_eff
from module_market import dis_fun, comove_d, growth_factor
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import integrate
from scipy import special
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Simulate lensing convergence power spectrum C^ij(l), made by Zhejie.', formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--nrbin", help = '*Number of tomographic bins.', type=int, required=True)
parser.add_argument("--nkout", help = '*Number of output k bins.', type=int, required=True)
parser.add_argument("--Pk_type", help = '*Type of P(k). Pwig_linear: linear power spectrum containing BAO; Pnow: Psm without BAO. Pwig_nonlinear: with BAO damped.', required=True)
parser.add_argument("--Psm_type", help = '*The expression of Pnorm. The default case, Pnorm from Eisenstein & Zaldarriaga 1999. \
                                          Test Pnorm=Pnow, which is derived from transfer function.')
parser.add_argument("--cal_sn", help = "Whether it's going to calculate (pseudo) shapenoise in each tomographic bin. True or False.")
parser.add_argument("--cal_cijl", help = "Whether it's going to calculate Cij(l). True or False.")
parser.add_argument("--cal_Gm", help = "Whether it's going to calculate G matrix. True or False")
parser.add_argument("--alpha", help = "BAO scale shifting parameter alpha.", default=1.0, type=float)
parser.add_argument("--odir0", help = '*The basic output directory, e.g., ./.', required=True)
parser.add_argument("--show_nz", help = "Whether it's going to plot n^i(z) distribution in photo-z tomographic bin. True or False.")

args = parser.parse_args()
nrbin = args.nrbin                      # nrbin represents the number of tomographic bins
num_kout = args.nkout                   # number of k output bins for output power spectrum
Pk_type = args.Pk_type                  # Simulate lensing power spectrum from 3D power spectrum P(k) with/without BAO.
Psm_type = args.Psm_type                # the expression of Pnorm, either the default case or Pnow from transfer function.
cal_sn = args.cal_sn
cal_cijl = args.cal_cijl
cal_Gm = args.cal_Gm
alpha = args.alpha
odir0 = args.odir0
show_nz = args.show_nz

#-------------------------------------------------------#
def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    t_0 = MPI.Wtime()
    survey_stage = 'KW_stage_III'
    N_dset = (nrbin+1)*nrbin//2            # numbe of C^ij(l) data sets
    #data_type_size = 8                     # number of bytes for double precison data
    zbin = np.zeros(nrbin+1)               # for both zbin and chibin, the first element is 0.
    chibin = np.zeros(nrbin+1)
    shape_noise = np.zeros(nrbin)

    num_kin = 506                             # the number of boundary points of k bins from the input matter power spectrum file
    # consider num_kbin as the input number of k bins
    num_kbin = num_kin - 1                    # k_max should be larger than lmax/xmax, -1 means disregarding the last term

    k_par = np.zeros(num_kbin)                # Input k and Pk for the calculation of C^ij(l)
    Pk_par = np.zeros(num_kbin)

    # l (the parameter of C^ij(l)) value equals to l_min, l_min+delta_l, ..., l_max-delta_l
    # We choose the case below:
    l_max = 2002                             # l_max < X_max*k_max
    l_min = 1
    delta_l = 3
    num_l = (l_max-l_min)//delta_l +1

    c = 2.99792458e5                        # speed of light unit in km/s
    H_0 = 100.0                               # unit: h * km/s/Mpc
    sigmae = 0.021                          # Tully-Fisher case \sigma_e from Eric's paper
    scale_n = 1.10                          # Tully-Fisher total surface number density (unit: arcmin^-2), from Eric et al.(2013), Table 2 (TF-Stage)

    cross_const = (1.5*cosmic_params.omega_m)**2.0*(H_0/c)**4.0  # It's the coefficent constant of convergence power spectrum, see Eq.(21)
    #print 'cross_const', cross_const
    sr_const = np.pi**2.0/1.1664e8          # 1 square acrminute = sr_const steradian
    constx = sr_const/cross_const           # The constx connects shot noise with C^ij(l)

    idir0 = '/Users/ding/Documents/playground/shear_ps/SVD_ps/'
    inputf = idir0 + 'Input_files/zdistribution_DES_Tully_Fisher.txt'             # Input file of n(z) which is the galaxy number density distribution in terms of z
    # Here center_z denotes z axis of n(z). It may not be appropriate since we don't have redshift bin setting
    lower_z, center_z, upper_z, n_z = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
    n_sum = np.sum(n_z*(upper_z - lower_z))                                          # Calculate the total number density
    #print(n_sum)
    scale_dndz = scale_n/n_sum
    n_z = n_z * scale_dndz                                                  # rescale n(z) to match the total number density from the data file equal to scale_n
    spl_nz = InterpolatedUnivariateSpline(center_z, n_z)                    # Interpolate n(z) in terms of z using spline

    #nz_test = interpolate.splev(center_z, tck_nz, der=0)
    #print(abs(n_z- nz_test)<1.e-7)

    # calculate total number density n^i (integrate dn/dz) in the ith tomographic bin
    def n_i_bin(zbin, i):
        zi = zbin[i]
        zf = zbin[i+1]
        # rescale n(z) to match the total number density from the data file equal to scale_n
        ##n_i = scale_dndz * integrate.quad(n_z, zi, zf, epsabs=1.e-7, epsrel=1.e-7)[0]
        n_i = spl_nz.integral(zi, zf)
        return n_i

    G_0 = growth_factor(0.0, cosmic_params.omega_m)         # G_0 at z=0, normalization factor
    num_z = np.size(center_z)                              # the number of z bins of n(z), obtained from the data file
    chi_z = np.zeros(num_z)

    for i in range(num_z):
        chi_z[i] = comove_d(center_z[i])*c/H_0           # with unit Mpc/h, matched with that of ell/k
    # we want interpolate z as a function of chi
    spl_zchi = InterpolatedUnivariateSpline(chi_z, center_z)  # z as a function of chi

    # here interpolate \chi as a function of z
    spl_chiz= InterpolatedUnivariateSpline(center_z, chi_z)

    # bin interval
    z_min = center_z[0]
    z_max = upper_z[-1]
    zbin_avg = (z_max-z_min)/float(nrbin)
    for i in range(nrbin):
        zbin[i]=i*zbin_avg + z_min
    zbin[-1]= z_max
    print("zbin:", zbin)
    # print('Xmax', c/H_0*comove_d(zbin[-1]))
    # print('nbar first element: ', n_i_bin(zbin, 0))

    # For stage-III, we set chibin[0] equal to 0, since the lowest z=0. Unit is Mpc/h.
    for i in range(1, nrbin+1):
        chibin[i] = comove_d(zbin[i])*c/H_0

    # 3D power spectrum is obtained from CAMB using the above cosmological parameters.
    ##inputf = fpath+'test_matterpower.dat'# if it's used, the cosmological parameters should also be changed correspondingly.
    inputf = idir0 + 'Input_files/CAMB_Planck2015_matterpower.dat'
    k_camb, Pk_camb = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
    Pk_camb_spl = InterpolatedUnivariateSpline(k_camb, Pk_camb)

    ifile = idir0 + 'Input_files/transfer_fun_Planck2015.dat'
    kk, Tf = np.loadtxt(ifile, dtype='f8', comments='#', usecols=(0,1), unpack=True)
    ##print(kk)
    k_0 = 0.001       # unit h*Mpc^-1
    Pk_0 = Pk_camb_spl(k_0)
    Tf_spl = InterpolatedUnivariateSpline(kk, Tf)
    Tf_0 = Tf_spl(k_0)
    P0_a = Pk_0/(pow(k_0, cosmic_params.ns) *Tf_0**2.0)
    Psm_transfer = P0_a * pow(k_camb, cosmic_params.ns) * Tf**2.0               # Get primordial (smooth) power spectrum from the transfer function
    Pk_now_spl = InterpolatedUnivariateSpline(k_camb, Psm_transfer)

    # ------ This part calculates the Sigma^2_{xy} using Pwig from CAMB. -------#
    z_mid = z_max/2.0
    q_BAO = 110.0   # unit: Mpc/h, the sound horizon scale
    Sigma2_integrand = lambda k: Pk_camb_spl(k) * (1.0 - np.sin(k * q_BAO)/(k * q_BAO))
    pre_factor = 1.0/(3.0 * np.pi**2.0)* (growth_factor(z_mid, cosmic_params.omega_m)/G_0)**2.0
    Sigma2_xy = pre_factor * integrate.quad(Sigma2_integrand, k_camb[0], k_camb[-1], epsabs=1.e-06, epsrel=1.e-06)[0]
    print('At z=', z_mid, 'Sigma2_xy=', Sigma2_xy)

    #----------------------------------------------------------------------------#
    def Pk_par_integrand(k):
        if Pk_type == 'Pwig_linear':
            Pk_par = Pk_camb_spl(k)
        elif Pk_type == 'Pnow':
            Pk_par = Pk_now_spl(k)
        elif Pk_type == 'Pwig_nonlinear':
            Pk_par = Pk_now_spl(k) + (Pk_camb_spl(k) - Pk_now_spl(k))* np.exp(-k**2.0*Sigma2_xy/2.0)
        return Pk_par

    odir1 = 'mpi_preliminary_data_{}/'.format(Pk_type)
    if Psm_type == 'Pnow':
        odir1_Gm = odir1 + 'set_Pnorm_Pnow/'
    else:
        odir1_Gm = odir1

    odir = odir0 + odir1 + 'comm_size{}/'.format(size)
    odir_Gm = odir0 + odir1_Gm + 'comm_size{}/'.format(size)

    if rank == 0:
        if not os.path.exists(odir):
            os.makedirs(odir)

        if not os.path.exists(odir_Gm):
            os.makedirs(odir_Gm)
    comm.Barrier()

    print('odir_Gm:', odir_Gm, 'from rank:', rank)
    prefix = 'Tully-Fisher'
    Cijl_outf_prefix = odir + prefix                                                # The prefix of output file name
    Gm_outf_prefix = odir_Gm + prefix
    iu1 = np.triu_indices(nrbin)                                              # Return the indices for the upper-triangle of an (n, m) array

    #------------------------------------------------
    def get_shapenoise(rank):
        if rank == 0:
            # Calculate covariance matrix of Pk, the unit of number density is per steradians
            for i in range(nrbin):
                shape_noise[i] = sigmae**2.0/n_i_bin(zbin, i)
            pseudo_sn = shape_noise*constx

            # Output the shape noise (includes the scale factor) in a file
            outf = odir0 + odir1 + prefix + '_pseudo_shapenoise_{0}rbins.out'.format(nrbin)              # basic variable
            np.savetxt(outf, pseudo_sn, fmt='%.15f', newline='\n')
    #---------------------------------------------------
    # Interpolate g^i in terms of chi (comoving distance)
    ifile = './lens_eff_g_i/g_i_{}rbins.npz'.format(nrbin)
    npz_data = np.load(ifile)
    chi_array = npz_data['chi_array'] * c/H_0
    g_i_matrix = npz_data['g_i_matrix']
    #print("g_i_matrix:", g_i_matrix)
    spl_gi_list = []
    for i in range(nrbin):
        spl_gi = InterpolatedUnivariateSpline(chi_array, g_i_matrix[i, :])
        spl_gi_list.append(spl_gi)

    ifile = idir0 + 'Input_files/{}_num_ell_per_rank_comm_size{}.dat'.format(survey_stage, size)
    num_ell_array = np.loadtxt(ifile, dtype='int', comments='#', usecols=(1,))
    num_l_in_rank = num_ell_array[rank]

    #------------------------------------------------------------------------------------------------#
    #--------------------------- Part 1: calculate C^ij(l) ------------------------------------------#
    # Note: Don't generate output G matrix for output P(k) in the process with C^ij(l), because the interval of k bins are
    # different from those of the G matrix for 'observered' data C^ij(l)!
    #------------------------------------------------------------------------------------------------#
    # This is for output G' matrix.
    def Gm_integrand_out(k, c_i, spl_gi, spl_gj, ell):
        chi_k = ell/k
        # Since the diameter of Milky Way is about 0.03 Mpc, we assume that the smallest interval between chi_k and chibin[i+1] larger than 0.1 Mpc/h.
        if (chibin[c_i+1]-1.e-8) < chi_k:
            return 0.0
        else:
            #z_k = interpolate.splev(chi_k, tck_zchi, der=0)
            z_k = spl_zchi(chi_k)
            GF = (growth_factor(z_k, cosmic_params.omega_m)/G_0)**2.0
            res = (1.0+z_k)**2.0* spl_gi(chi_k) * spl_gj(chi_k) * ell/k**2.0 * GF
            return res

    # This is for output C^{ij}(l).
    def Gm_integrand_in(k, c_i, spl_gi, spl_gj, ell):
        return Gm_integrand_out(k, c_i, spl_gi, spl_gj, ell) * Pk_par_integrand(k)

    def get_Cijl(comm, rank):
        # Output the Cij_l array in which each term is unique.
        def cal_cijl(l, rank):
            #n_l = default_num_l_in_rank * rank + l
            n_l = np.sum(num_ell_array[0: rank]) + l
            ell = l_min + n_l * delta_l
            ell = ell * alpha
            #offset_cijl = n_l * N_dset * data_type_size
            c_temp = np.zeros((nrbin, nrbin))
            for c_i in range(nrbin):
                for c_j in range(c_i, nrbin):
                    # we could use smaller epsrel, but it would require more integration points to achieve that precision.
                    res = integrate.quad(Gm_integrand_in, k_camb[0], k_camb[-1], args=(c_i, spl_gi_list[c_i], spl_gi_list[c_j], ell), limit=200, epsabs=1.e-6, epsrel=1.e-12)
                    c_temp[c_i][c_j] = res[0]
                    abserr = res[1]
                    if res[0] != 0.0:
                        relerr = abserr/res[0]
                    else:
                        relerr = 0.0
                #c_temp[c_i][c_i : nrbin] = np.dot(gmatrix_jk, Pk_par)
            array_cij = np.asarray(c_temp[iu1], dtype=np.float64)  # extract upper-triangle of c_temp
            if rank == 0:
                #print('rank:', rank, 'array_cij:', array_cij)
                print('ell from rank', rank, 'is', ell, 'abs err of Cijl is %.4e'%abserr, 'and rel err is %.4e'%relerr)
            return ell, array_cij, abserr, relerr


        Cijl_file = Cijl_outf_prefix + '_Cij_l_{0}rbins_{1}kbins_CAMB_rank{2}.bin'.format(nrbin, num_kbin, rank)                # basic variable
        Cijl_fwriter = open(Cijl_file, 'wb')

        err_info = np.array([], dtype=np.float64).reshape(0, 3)
        for l in range(num_l_in_rank):
            ell, cijl, abserr, relerr = cal_cijl(l, rank)
            cijl.tofile(Cijl_fwriter, sep="")
            err_info = np.vstack((err_info, np.array([ell, abserr, relerr])))
        Cijl_fwriter.close()
        err_info_ofile = Cijl_outf_prefix + '_integration_error_Cij_l_{0}rbins_{1}kbins_CAMB_rank{2}.out'.format(nrbin, num_kbin, rank)
        np.savetxt(err_info_ofile, err_info, fmt='%i  %.4e  %.4e', delimiter=' ', newline='\n', header='ell    abs_err   rel_err', comments='#')
        #comm.Barrie()

    #-----------------------------------------------------------------------------------------------#
    #------------------------- Part 2: get Gm_cross_out for output P(k) ----------------------------#

    ######------------- set up output k space and G' matrix for output Pk ----------------###########
    def get_Gm_out(comm, rank):
        # construct Gmatrix: Gout for output Pk with num_kout kbins
        # Note: The algorithm is the same as that calculating C^ij(l) in Part 1. Here we use a simplified (looks like) way to get Gmatrix_l.
        def cal_G(l, rank):
            n_l = np.sum(num_ell_array[0: rank]) + l
            ell = l_min + n_l * delta_l
            ell = ell * alpha
            #offset_Gm = n_l * N_dset * num_kout * data_type_size

            Gmatrix_l = np.zeros((N_dset, num_kout))
            # j denotes column
            for j in range(num_kout):
                # i denotes row
                for i in range(N_dset):
                    # redshift bin i: rb_i
                    rb_i = iu1[0][i]
                    # in python, eps should be larger than 1.e-15, to be safe. The smallest chi from the corresponding output k bin should be smaller than the
                    # the upper boundary of chi from the ith tomographic bin
                    if chibin[rb_i+1] > ell/kout[j+1]:
                        ##krb_i = ell/(chibin[rb_i]+1.e-12)  # avoid to be divided by 0
                        rb_j = iu1[1][i]
                        # more precise calculation of Gmatrix_l
                        # the j index of Pnorm_out denotes k bin id, different from the index rb_j of g_j
                        res = integrate.quad(Gm_integrand_out, kout[j], kout[j+1], args=(rb_i, spl_gi_list[rb_i], spl_gi_list[rb_j], ell), limit=200, epsabs=1.e-6, epsrel=1.e-12)
                        Gmatrix_l[i][j] = res[0] * Pnorm_out[j]
                        abserr = res[1]
                        if res[0] != 0.0:
                            relerr = abserr/res[0]
                        else:
                            relerr = 0.0
            #print Gmatrix_l[:, 0]
            if rank == 0:
                #print('rank:', rank, 'Gm:', Gmatrix_l)
                print('ell from rank', rank, 'is', ell, 'abs err of G is %.4e'%abserr, 'rel err is %.4e'%relerr)
            return ell, Gmatrix_l, abserr, relerr

        kout, k_mid = np.zeros(num_kout+1), np.zeros(num_kout)
        k_low, k_high = 0.01, 1.0                              # This set may need to be checked more!
        kout[0], kout[1], kout[-1] = k_camb[0], k_low, k_camb[-1]
        lnk_factor = np.log(k_high/k_low)/(num_kout-2)

        for i in range(2, num_kout):
            kout[i] = kout[i-1]*np.exp(lnk_factor)
        #print kout

        for i in range(num_kout):
            k_mid[i] = (kout[i] + kout[i+1])/2.0
        if Psm_type == 'Pnorm' or Psm_type == 'default':
            Pnorm_out = 1.5e4/(1.0+(k_mid/0.05)**2.0)**0.65                                       # from Eisenstein & Zaldarriaga (2001)
        elif Psm_type == 'Pnow':
            Pnorm_out = Pk_now_spl(k_mid)              # Test how the change of Pnow could influence the eigenvalues from SVD routine.

        # Gm_cross_out uses selected new k bins
        Gm_cross_file = Gm_outf_prefix + '_Gm_cross_out_{0}rbins_{1}kbins_CAMB_rank{2}.bin'.format(nrbin, num_kout, rank)         # basic variable
        Gm_cross_fwriter = open(Gm_cross_file, 'wb')

        err_info = np.array([], dtype=np.float64).reshape(0, 3)
        for l in range(num_l_in_rank):
            ell, Gm, abserr, relerr = cal_G(l, rank)
            Gm.tofile(Gm_cross_fwriter, sep="")
            err_info = np.vstack((err_info, np.array([ell, abserr, relerr]))) # If relerr = xx/0, it doesn't seem to be appendable on array.
        Gm_cross_fwriter.close()
        err_info_ofile = Gm_outf_prefix + '_integration_error_Gm_cross_out_{0}rbins_{1}kbins_CAMB_rank{2}.out'.format(nrbin, num_kbin, rank)
        np.savetxt(err_info_ofile, err_info, fmt='%i  %.4e  %.4e', delimiter=' ', newline='\n', header='ell    abs_err   rel_err', comments='#')


    if cal_sn == "True":
        get_shapenoise(rank)
    if cal_cijl == "True":
        get_Cijl(comm, rank)
    #comm.Barrier()
    t_1 = MPI.Wtime()

    if cal_Gm == "True" and Pk_type == 'Pwig_nonlinear':
        get_Gm_out(comm, rank)
    #comm.Barrier()
    t_2 = MPI.Wtime()
    if rank == 0:
        print('Running time for Cijl:', t_1-t_0)
        print('Running time for G matrix:', t_2-t_1)

#######################################################

    def plot_numd_spectroscopy():
        odir_data = "./numd_distribute_spectro/"
        if not os.path.exists(odir_data):
            os.makedirs(odir_data)
        odir_fig = odir_data + 'nz_fig/'
        if not os.path.exists(odir_fig):
            os.makedirs(odir_fig)
        nd_avg = []
        for i in range(nrbin):
            nd_avg.append(n_i_bin(zbin, i)/(zbin[i+1]-zbin[i]))
        ofile = odir_data + 'gal_numden_spectroz_{}rbins.out'.format(nrbin)
        header_line = ' bin_boundary(low)   nz_avg'
        np.savetxt(ofile, np.array([zbin[0:-1], nd_avg]).T, fmt='%.7f', newline='\n', comments='#')

        print("nd_avg:", nd_avg, "zbin:", zbin)
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(left=zbin[0:-1], height=nd_avg, width=zbin_avg, align='edge', color='white', edgecolor='grey')
        bars[11].set_color('r')
        print(bars)
        # n, bins, pathes = ax.hist(nd_avg, bins=nrbin, range=[zbin[0], zbin[-1]], align='left')
        # print(n, bins, pathes)
        ax.plot(center_z, n_z, 'k-', lw=2.0)
        ax.set_xlim([0.0, z_max])
        ax.set_ylim([0.0, 1.8])
        ax.set_xlabel(r'$z$', fontsize=20)
        #ax.set_ylabel('$n^i(z)$ $[\mathtt{arcmin}]^{-2}$', fontsize=20)
        ax.set_ylabel(r'$dn^i/dz \; [\mathtt{arcmin}]^{-2}$', fontsize=20)
        ax.minorticks_on()
        ax.tick_params('both', length=5, width=2, which='major', labelsize=15)
        ax.tick_params('both', length=3, width=1, which='minor')
        ax.set_title("KWL-Stage III", fontsize=20)

        plt.tight_layout()
        figname = "gal_numden_{}rbins_spectro.pdf".format(nrbin)
        plt.savefig(odir_fig + figname)
        plt.show()
        plt.close()

    if show_nz == "True" and rank == 0:
        plot_numd_spectroscopy()

if __name__ == '__main__':
    main()

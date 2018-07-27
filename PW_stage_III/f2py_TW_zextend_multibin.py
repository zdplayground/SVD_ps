#!/Users/ding/anaconda3/bin/python
# --------------------------------------------------
# Copy the f2py_TW_zextend_multibin.py from ../pseudo_PW_stage_IV, modify the \sigma_z to be 0.1(1+z) from DES. --03/19/2018
# Note that \sigma_z is a parameter of photo-z distribution given the true redshift. We need to modify function photo_ni.
# Here we set n_gal=31 as the default case. If we need to set it to be 10 instead, we could set the effective shape noise factor in code mpi_PW_sn_dep_Cov_multibin.py.
# 1. Add the input of the number of ell (work) need to be executed for each rank based on the approximated calculation time for low and high ells. --06/14/2018
# 2. Add the angurement alpha as the BAO scale shifting parameter, it is applied on ell such as ell' = alpha * ell. --07/27/2018
#
#######################################################################################################################################################
# This program is based on f2py_TW_zextend, modified to generate .bin data files, 07/23/2016. (Retain the notes from the previous code.)
# We use f2py to wrap fortran subroutines for lens efficiency function calculation. Generate the module: lens_eff_module
# Firstly copied: 11/09/2015
# modified: 11/11/2015: change n_gal(scale_n) from 1.1 to 10.0
# modified: 12/09/2015, change k_par dimension from num_par to num_par+1, path of data files
# modified: 01/08/2016, use cosmological parameters from Planck 2015, and the corresponding power spectrum from CAMB
# modified: 01/28/2016, extend zmax to 1.64, do 100 averaged bins in z<1.3 for spectroscopic bins, do more than 100 bins in z<1.65 for photo-z bins,
# note that information of 100 spectroscopic bins is from the whole photo-z bins
# 04/14/2016, modify Gout matrix to match with that in the T-F case.
# 04/17/2016, rename the code name from f2py_test7_zextend.py to f2py_TW_zextend.py; rename pseudo shapenoise file as '..rbins_ext..', ext means
# including the external spectroscopic bins over num_rbin.
# 04/24/2016, find the Gout previously set was wrong, we need to include cross terms as well, modified it.
# 04/28/2016, examined the code is correct.
# 05/02/2016, for a large number of tomographic bins, e.g. 100, need to modify the writing out process for Gm_cross_out, see reference in sn_dep_Cov_Cij.py
# 06/15/2016, add output ni_zp from the function photo_ni; output ni_zp for 5 tomographic bins for my poster 2017, -- 04/02/2017.
# 07/21/2016, rename the shapenoise output file and tidy the code.
# 08/16 - 09/01/2017, modify it for Python3 version.
# 09/06/2017, include BAO damping in input power spectrum.
#
from mpi4py import MPI
import numpy as np
import math
import time
import os, sys
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/')
import cosmic_params
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import integrate
from scipy import special
from lens_eff_module import lens_eff, photo_ni, comoving_d, gauleg, spline, splint
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Use mcmc routine to get the BAO peak stretching parameter alpha and damping parameter, made by Zhejie.')
parser.add_argument("--nrbin", help = '*Number of tomographic bins.', type=int, required=True)
parser.add_argument("--nkout", help = '*Number of output k bins.', type=int, required=True)
parser.add_argument("--Pk_type", help = "*The input power spectrum. Pwig_linear: linear power spectrum with BAO wiggles; Pwig_nonlinear: BAO wiggles damped. \
                    Pnow: nowiggle power spectrum.", required=True)
parser.add_argument("--cal_sn", help = "Whether it's going to calculate (pseudo) shapenoise in each tomographic bin. True or False.")
parser.add_argument("--cal_cijl", help = "Whether it's going to calculate Cij(l). True or False.")
parser.add_argument("--cal_Gm", help = "Whether it's going to calculate G matrix. True or False")
parser.add_argument("--show_nz", help = "Whether it's going to plot n^i(z) distribution in photo-z tomographic bin. True or False.")
parser.add_argument("--odir0", help = "The basic directory for output files, default is './'.", required=True)
parser.add_argument("--alpha", help = "The BAO scale shifting parameter alpha, i.e. k'=alpha * k or l'= alpha * l given a \chi. ", default=1.0, type=np.float64)

##parser.add_argument("--num_eigv", help = 'Number of eigenvalues included from SVD.', required=True)
args = parser.parse_args()

num_rbin = args.nrbin                 # for photo-z<1.3
num_kout = args.nkout                 # number of kbins for output power spectrum, which corresponds to num_par
Pk_type = args.Pk_type                # The type of input power spectrum
cal_sn = args.cal_sn
cal_cijl = args.cal_cijl
cal_Gm = args.cal_Gm
show_nz = args.show_nz
alpha = args.alpha
#----------------------------------------------------------------------------------------#

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    num_kin = 506                         # number of k point in the input matter power spectrum file
    # l value from l_min, l_min+delta_l, ..., l_max-delta_l
    l_max = 2002                          # lmax<Xmax*k_max
    l_min = 1
    delta_l = 3
    num_l = (l_max-l_min)//delta_l +1
    # index of kmin from CAMB
    kpar_min = 0
    # k_max should be larger than lmax/xmax. -1 means disregarding the last term. Basically it's the number of k bins from input.
    num_par = num_kin - kpar_min - 1

    # dimensin of parameter vector
    k_par = np.zeros(num_par)
    Pk_par = np.zeros(num_par)
    Pnorm = np.zeros(num_par)

    c = 2.99792458e5      # speed of light unit in km/s

    ##----- in traditational WL, sigmae is larger, equal to 0.26 from Table 2 in Eric et al. 2013 ----#
    sigmae = 0.26
    ####scale_n = 10.0
    scale_n = 31.0

    # In this code, the constant of cross power spectrum is
    cross_const = (1.5*cosmic_params.omega_m)**2.0*(100/c)**4.0
    #print 'cross_const', cross_const
    # 1 square acrminute = sr_const steradian
    sr_const = np.pi**2.0/1.1664e8
    constx = sr_const/cross_const

    idir0 = '/Users/ding/Documents/playground/shear_ps/SVD_ps/'
    # input galaxy number density n(z) file
    inputf = idir0 + 'Input_files/zdistribution_DES_Tully_Fisher.txt'
    lower_z, center_z, upper_z, n_z = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)

    #print n_z[-1]~ 0.02 at z~1.3
    n_sum = sum(n_z*(upper_z - lower_z))
    #print(n_sum)
    scale_dndz = scale_n/n_sum
    ##--consider n_z as the overall galaxy distribution defined in equation(2) in Ma, Z et al. 2008---##
    n_z *= scale_dndz
    ## spline n_z as a function of z
    tck_nz = interpolate.splrep(center_z, n_z)
    zmax = center_z[-1]
    n_p_zmax = interpolate.splev(zmax, tck_nz, der=1)    # the first derivative at z_max~1.3

    # fitting function nc1 * exp(nc2*z)
    #nc2 = n_p_zmax/n_z[-1]
    #nc1 = n_z[-1]/math.exp(nc2 * center_z[-1])

    # fitting function using nc1*z^2*exp(-z^2/nc2), Ma, Z. et al. 2006
    nc2 = 2.0* zmax**2.0 *n_z[-1]/(2.0*n_z[-1]-zmax*n_p_zmax)
    nc1 = n_z[-1]/(zmax**2.0 * math.exp(-zmax**2.0/nc2))
    #print("z_0^2: ", nc2, "Const: ", nc1)

    # set zmax as the maximum z after extension
    zmax_ext = zmax + math.sqrt(2.0)*0.05*(1+zmax)*2.2   # 2.2 is from erf(2.2) which is close to 1.0
    print('zmax_ext: ', zmax_ext)   # It's about 1.655.
    # add 0.001 which is just a small number, so zext[0] doesn't coincide with center_z[-1]
    zext = np.linspace(center_z[-1]+0.001, zmax_ext, 80) # extended z region for the external tomographic bins
    nzext = nc1* zext**2.0 *np.exp(-zext**2.0/nc2)

    #print('n(zmax):', nzext[-1]) # it's around 1.64
    center_z = np.append(center_z, zext)
    n_z = np.append(n_z, nzext)

    num_z = len(center_z)                                # set the new num_z
    chi_z = np.zeros(num_z)
    nz_y2 = np.zeros(num_z)

    tck_nz = interpolate.splrep(center_z, n_z)           # do a new spline interpolation

    ##--for traditational WL, the total distribution of galaxies n(z) in tomographic bins is unchanged regardless
    ## how complicate the photo-z probability distribution is --#
    # calculate the number density nbar^i (integrate dn/dz) in the ith tomographic bin
    # the unit of number density is per steradian
    def n_i_bin(i):
        zi = zbin[i]
        zf = zbin[i+1]
        n_i = interpolate.splint(zi, zf, tck_nz)
        return n_i

    # it's from expression 1/H(z) without the constant
    def dis_fun(z):
        return 1.0/math.sqrt(cosmic_params.omega_m*(1.0+z)**3.0+ cosmic_params.omega_L)

    # without c/H_0 constant
    def comove_d(z):
        distance = integrate.quad(dis_fun, 0.0, z)
        return distance[0]

    ##-----set tomographic bins-------------##
    for i in range(num_z):
        chi_z[i] = comove_d(center_z[i])*c/100.0
    # we want interpolate z as a function of chi
    tck_zchi = interpolate.splrep(chi_z, center_z)

    tck_chiz= interpolate.splrep(center_z, chi_z)

    # bin interval
    zbin_avg = zmax/float(num_rbin) # use zmax as the center_z[-1] before z extension
    nbin_ext = int(zmax_ext/zbin_avg)
    #print('# of redshift bins (extended): ', nbin_ext)

    # for zbin and chibin, the first element is 0.
    zbin = np.zeros(nbin_ext+1)
    chibin = np.zeros(nbin_ext+1)
    for i in range(nbin_ext+1):
        zbin[i]=float(i)*zbin_avg
        # just note: chibin[0] is 0, chibin[1] stores the first bin's up boundary
        chibin[i] = interpolate.splev(zbin[i], tck_chiz, der=0)

    chibin[0] = 0.0 # reset chibin[0] equal to 0 when z=0

    #print('Xmax', c/100.0*comove_d(zbin[-1]))
    #print('Zmax_ext', zbin[-1])


    # define growth factor G(z)
    def growth_factor(z, Omega_m):
        a = 1.0/(1.0+z)
        v = (1.0+z)*(Omega_m/(1.0-Omega_m))**(1.0/3.0)
        phi = math.acos((v+1.0-3.0**0.5)/(v+1.0+3.0**0.5))
        m = (math.sin(75.0/180.0* math.pi))**2.0
        part1c = 3.0**0.25 * (1.0+ v**3.0)**0.5
    # first elliptic integral
        F_elliptic = special.ellipkinc(phi, m)
    # second elliptic integral
        Se_elliptic = special.ellipeinc(phi, m)
        part1 = part1c * ( Se_elliptic - 1.0/(3.0+3.0**0.5)*F_elliptic)
        part2 = (1.0 - (3.0**0.5 + 1.0)*v*v)/(v+1.0+3.0**0.5)
        d_1 = 5.0/3.0*v*(part1 + part2)
    # if a goes to 0, use d_11, when z=1100, d_1 is close to d_11
    #    d_11 = 1.0 - 2.0/11.0/v**3.0 + 16.0/187.0/v**6.0
        return a*d_1
    G_0 = growth_factor(0.0, cosmic_params.omega_m) # G_0 at z=0, normalization factor


    # 3D power spectrum is from CAMB
    ##inputf = '../test_matterpower.dat'
    inputf = idir0 + 'Input_files/CAMB_Planck2015_matterpower.dat'
    k_camb, Pk_camb = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
    Pk_camb_spl = InterpolatedUnivariateSpline(k_camb, Pk_camb)

    ifile = idir0 + 'Input_files/transfer_fun_Planck2015.dat'
    kk, Tf = np.loadtxt(ifile, dtype='f8', comments='#', usecols=(0,1), unpack=True)
    k_0 = 0.001       # unit h*Mpc^-1
    Pk_0 = Pk_camb_spl(k_0)
    Tf_spl = InterpolatedUnivariateSpline(kk, Tf)
    Tf_0 = Tf_spl(k_0)
    P0_a = Pk_0/(pow(k_0, cosmic_params.ns) *Tf_0**2.0)
    Psm_transfer = P0_a * pow(k_camb, cosmic_params.ns) * Tf**2.0               # Get primordial (smooth) power spectrum from the transfer function
    Pk_now_spl = InterpolatedUnivariateSpline(k_camb, Psm_transfer)

    # ------ This part calculates the Sigma^2_{xy} using Pwig from CAMB. -------#
    z_mid = upper_z[-1]/2.0
    q_BAO = 110.0   # unit: Mpc/h, the sound horizon scale
    Sigma2_integrand = lambda k: Pk_camb_spl(k) * (1.0 - np.sin(k * q_BAO)/(k * q_BAO))
    pre_factor = 1.0/(3.0 * np.pi**2.0)* (growth_factor(z_mid, cosmic_params.omega_m)/G_0)**2.0
    Sigma2_xy = pre_factor * integrate.quad(Sigma2_integrand, k_camb[0], k_camb[-1], epsabs=1.e-03, epsrel=1.e-03)[0]
    #print('At z=', z_mid, 'Sigma2_xy=', Sigma2_xy)

    for i in range(num_par):
        k_par[i] = (k_camb[i]+k_camb[i+1])/2.0
        # We didn't include 'now' type here as was did in Tully-Fisher case.
        if Pk_type == 'Pwig_linear':
            Pk_par[i] = Pk_camb_spl(k_par[i])
        elif Pk_type == 'Pnow':
            Pk_par[i] = Pk_now_spl(k_par[i])
        elif Pk_type == 'Pwig_nonlinear':
            Pk_par[i] = Pk_now_spl(k_par[i]) + (Pk_camb_spl(k_par[i]) - Pk_now_spl(k_par[i]))* np.exp(-k_par[i]**2.0*Sigma2_xy/2.0)

    odir0 = args.odir0
    if alpha != None:
        odir0 = odir0 + 'BAO_alpha_{}/'.format(alpha)
    odir = odir0 + 'mpi_preliminary_data_{}/comm_size{}/'.format(Pk_type, size)
    if rank == 0:
        if not os.path.exists(odir):
            os.makedirs(odir)
    outf_prefix = odir + 'TW_zext_'

    def get_shapenoise():
        shape_noise = np.zeros(nbin_ext)
        # Calculate covariance matrix of Pk, the unit of number density is per steradians
        for i in range(nbin_ext):
            shape_noise[i] = sigmae**2.0/n_i_bin(i)
        #shape_noise[i] = sigmae**2.0/ s_nz[i] # It's the serious bug that I made and couldn't find it for half a year!
        pseudo_sn = shape_noise*constx

        # put the shape noise (includes the scale factor) in a file
        outf = outf_prefix + 'pseudo_shapenoise_{0}rbins_ext.out'.format(nbin_ext)                   # basic variable
        np.savetxt(outf, pseudo_sn, fmt='%.15f', newline='\n')


    ifile = idir0 + 'Input_files/PW_stage_num_ell_per_rank_comm_size{}.dat'.format(size)  # num of ell is roughly estimated
    num_ell_array = np.loadtxt(ifile, dtype='int', comments='#', usecols=(1,))
    num_l_in_rank = num_ell_array[rank]

    data_type_size = 8
    N_dset = (nbin_ext+1)*nbin_ext//2
    iu1 = np.triu_indices(nbin_ext)
    #
    ##################################################################################################################
    #------------------------------------ get cross power spectrum C^ij(l) ------------------------------------------#
    ##################################################################################################################
    # If we use an up-triangle matrix to store C^ij(l) with nbin_ext redshift bins, it's easier to extract C^ij(l) within numb_bin spectroscopic bins.
    # But to save space, I used array form to store C^ij(l) for each l.
    def get_Cijl(comm, rank):
        def cal_cijl(l, rank):
            n_l = np.sum(num_ell_array[0: rank]) + l
            ell = l_min + n_l * delta_l
            ell = alpha * ell
            ##offset_cijl = n_l * N_dset * data_type_size
            c_temp = np.zeros((nbin_ext, nbin_ext))
            for c_i in range(nbin_ext):
                g_nr = nbin_ext - c_i
                #print('g_nr:', g_nr)
                # gmatrix_jk is used to store integration elements to calculate array C^ij(l) from Pk for a certain l
                # j=i, i+1,...nbin_ext; j in the name gmatrix_jk also denotes row index
                gmatrix_jk = np.zeros((g_nr, num_par))
                #print(gmatrix_jk)
                # g_col is the column index of gmatrix_jk
                for g_col in range(num_par):
                    chi_k = ell/k_par[g_col]
                    z_k = interpolate.splev(chi_k, tck_zchi, der=0)   # Here z_k is understood as the spectroscopic redshift z
                    g_i = lens_eff(zbin, center_z, n_z, nz_y2, c_i, z_k)
                    if z_k < zmax:  # zmax corresponding to \chi_h in the expression of C^ij(l)
                        GF = (growth_factor(z_k, cosmic_params.omega_m)/G_0)**2.0
                        #print('zmax, z_k, GF:', zmax, z_k, GF)
                        c_j = c_i
                        # here g_row is the row index of gmatrix_jk
                        for g_row in range(g_nr):
                            g_j = lens_eff(zbin, center_z, n_z, nz_y2, c_j, z_k)
                            gmatrix_jk[g_row][g_col] = pow((1.0+z_k), 2.0)* g_i * g_j *ell*(1.0/k_camb[g_col]-1.0/k_camb[g_col+1])*GF
                            ###gmatrix_jk[g_row][g_col] = pow((1.0+z_k), 2.0)*lens_eff(c_i, chi_k)*lens_eff(c_j, chi_k)*ell*(1.0/k_par[g_col]-1.0/k_par[g_col+1])*GF
                            c_j += 1

                c_temp[c_i][c_i:nbin_ext] = np.dot(gmatrix_jk, Pk_par)
            #    print(c_temp)
            cijl = np.asarray(c_temp[iu1], dtype=np.float64)  # extract upper-triangle of c_temp
            if rank == 0:
                print('ell from rank', rank, 'is', ell)
            return cijl

        #eps = zbin_avg/10.0              # set minimum interval for g^i integration, z_max-z>eps  !maybe I need to consider this, 04/25/2016
        # Output the Cij_l array in which each term is unique.
        Cijl_file = outf_prefix + 'Cij_l_{}rbins_ext_{}kbins_CAMB_rank{}.bin'.format(nbin_ext, num_par, rank)                # basic variable
        # open file, write and append data in binary format (save storing volume)
        Cijl_fwriter = open(Cijl_file, 'wb')

        for l in range(num_l_in_rank):
            cijl = cal_cijl(l, rank)
            cijl.tofile(Cijl_fwriter, sep="")
        Cijl_fwriter.close()


    ##################################################################################################################
    #####################-------------------get output k space and G' matrix for output Pk-----------#################
    ##################################################################################################################
    def get_Gm_out(comm, rank):
        # odir_list = ['./Gm_cross_out_linear_k_data/', './Gm_cross_out_exp_k_data/']
        case = 1  # output k exponentially distributed
        kout, k_mid, Pnorm_out = np.zeros(num_kout+1), np.zeros(num_kout), np.zeros(num_kout)
        # Try k bins linearly distributed
        if case == 0:
            delta_k = (k_camb[-1]-k_camb[0])/num_kout
            kout[0] = k_camb[0]
            for i in range(num_kout):
               kout[i+1] = kout[i]+delta_k
               k_mid[i] = (kout[i+1]+kout[i])/2.0
               Pnorm_out[i] = 1.5e4/(1.0+(k_mid[i]/0.05)**2.0)**0.65

        # Try the simplest case, using the trapezoidal rule. Try to use exponentially distributed k bins
        elif case == 1:
            k_low, k_high = 0.01, 1.0
            kout[0], kout[1], kout[-1] = k_camb[0], k_low, k_camb[-1]
            lnk_factor = np.log(k_high/k_low)/(num_kout-2)

            for i in range(2, num_kout):
                kout[i] = kout[i-1]*np.exp(lnk_factor)
            #print(kout)
            for i in range(num_kout):
                k_mid[i] = (kout[i] + kout[i+1])/2.0
                Pnorm_out[i] = 1.5e4/(1.0+(k_mid[i]/0.05)**2.0)**0.65     # This Pnorm is from Eisenstein & Zaldarriaga 1999.

        # construct Gmatrix: Gout for output Pk with num_kout kbins
        def cal_Gm(l, rank):
            n_l = np.sum(num_ell_array[0: rank]) + l
            ell = l_min + n_l * delta_l
            ell = alpha * ell
            ##offset_Gm = n_l * N_dset * num_kout * data_type_size

            Gmatrix_l = np.zeros((N_dset, num_kout))
            # j denotes column
            for j in range(num_kout):
                #chi_k: comoving distance from k
                chi_k = ell/k_mid[j]         # I would have to say I could only do approximation here, e.g., using k_mid
                z_k = interpolate.splev(chi_k, tck_zchi, der=0)
                # i denotes row
                if z_k < zmax:
                    GF = (growth_factor(z_k, cosmic_params.omega_m)/G_0)**2.0
                    for i in range(N_dset):
                        # redshift bin i: rb_i
                        rb_i = iu1[0][i]
                        gi = lens_eff(zbin, center_z, n_z, nz_y2, rb_i, z_k)
                        # redshift bin j: rb_j
                        rb_j = iu1[1][i]
                        gj = lens_eff(zbin, center_z, n_z, nz_y2, rb_j, z_k)
                        # here too, I did approximation for the integration, e.g., the term (1/k1 - 1/k2)
                        Gmatrix_l[i][j] = pow((1.0+z_k), 2.0)* gi * gj *ell*(1.0/kout[j]-1.0/kout[j+1])*GF*Pnorm_out[j]
            return Gmatrix_l

        # Gm_cross_out uses selected new k bins
        Gm_cross_file = outf_prefix + 'Gm_cross_out_{}rbins_{}kbins_CAMB_rank{}.bin'.format(nbin_ext, num_kout, rank)         # basic variable
        Gm_cross_fwriter = open(Gm_cross_file, 'wb')
        for l in range(num_l_in_rank):
            Gm = cal_Gm(l, rank)
            Gm.tofile(Gm_cross_fwriter, sep="")
        Gm_cross_fwriter.close()


    if cal_sn == "True" and rank == 0:
        get_shapenoise()

    time0 = MPI.Wtime()
    if cal_cijl == "True":
        get_Cijl(comm, rank)
    time1 = MPI.Wtime()
    if rank == 0:
        print('Running time for Cijl:', time1 - time0)
    if Pk_type != 'Pnow' and cal_Gm == "True":
        get_Gm_out(comm, rank)
        time2 = MPI.Wtime()
        if rank == 0:
            print('Running time for Gm:', time2 - time1)

#######################################################
    #
    #---------- output galaxy number density distribution in each photo-z bin --------------#
    #
    def get_photoz_density(center_z):
        #print(center_z)
        odir = "./numd_distribute_photoz/" # output number density distribution in each tomographic bins.
        if not os.path.exists(odir):
            os.makedirs(odir)
        photoz_matrix = np.array([], dtype='float64').reshape(0, num_z)
        for bin_id in range(nbin_ext):
            photoz_array = np.zeros(num_z)
            i = 0
            for z_sp in center_z:
                ni_zp = photo_ni(zbin, center_z, n_z, nz_y2, bin_id, z_sp)  # function photo_ni function outputs n_i(z) at redshift z
                photoz_array[i] = ni_zp
                i = i+1
            photoz_matrix = np.vstack([photoz_matrix, photoz_array])

        ofile = odir + "gal_numden_photoz_{}rbins.out".format(nbin_ext)
        header_line = " z(spectroscopic)   n_i(z) for each ith photo-z bin"  # first column is z, the other columns are n_i(z) with increasing bin id
        np.savetxt(ofile, np.vstack([center_z, photoz_matrix]).T, fmt='%.7e', header=header_line, newline='\n', comments='#')

    def plot_numd_photoz():
        idir = "./numd_distribute_photoz/"
        odir = idir + "nz_fig/"
        if not os.path.exists(odir):
            os.makedirs(odir)

        ifile = idir + "gal_numden_photoz_{}rbins.out".format(nbin_ext)
        #-- simply show the results ---#
        data_m = np.loadtxt(ifile, dtype='f8', delimiter=' ', comments='#')
        center_z = data_m[:, 0]
        photoz_matrix = data_m[:, 1:]
        #print(photoz_matrix)
        fig, ax = plt.subplots()
        for bin_id in range(nbin_ext):
            if bin_id == 11:  # Show the 12th tomographic bin, which has the largest n^i(z)
                ax.plot(center_z, photoz_matrix[:, bin_id], 'r-.', lw=1.0)
            else:
                ax.plot(center_z, photoz_matrix[:, bin_id], 'k-.', lw=0.5)
            ax.axvline(zbin[bin_id+1], color='grey', linestyle=':')
        ax.plot(center_z, n_z, 'k-', lw=2.0)
        ax.set_xlabel('$z$', fontsize=20)
        ax.set_xlim([0.0, zmax])
        ax.set_ylim([0.0, 50])
        ax.minorticks_on()
        ax.tick_params('both', length=5, width=2, which='major', labelsize=15)
        ax.tick_params('both', length=3, width=1, which='minor')

        ax.set_ylabel('$n^i(z)$ $[\mathrm{arcmin}]^{-2}$', fontsize=20)
        ax.set_title("PW Stage-III", fontsize=18)
        plt.tight_layout()
        figname = "gal_numden_{}rbins_photoz.pdf".format(nbin_ext)
        plt.savefig(odir + figname)
        plt.show()

    if show_nz == "True" and rank == 0:
        get_photoz_density(center_z)
        plot_numd_photoz()


if __name__ == '__main__':
    main()

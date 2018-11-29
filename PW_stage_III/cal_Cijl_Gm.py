#!/Users/ding/anaconda3/bin/python
# Copy the code f2py_TW_zextend_multibin.py from ../PW_stage_IV/, modify the input file of source galaxy distribution, total angular number density (10/arcmin^2),
# sigma_z (photo-z uncertainty). --09/20/2018
# 1. Modify the function plot_numd_photoz, set dotted lines starting from y=0 to y=n(z). Correct the amplitude of n(z) in the figure. Note that the bug doesn't affect the
#    lensing power spectrum calculated before.  --11/25/2018
from mpi4py import MPI
import numpy as np
import math
import os, sys
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/')
import cosmic_params
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import integrate
from scipy import special
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/PW_modules/')
from lens_eff_module import lens_eff, photo_ni
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/common_modules/')
from module_market import dis_fun, comove_d, growth_factor
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Calculate the convergence power spectrum and the G matrix, made by Zhejie.')
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
    # For PW-Stage III, the constant of sigma_z (systematic error of source redshift from photometry)
    sigma_z_const = 0.1
    scale_n = 10.0

    # In this code, the constant of cross power spectrum is
    cross_const = (1.5*cosmic_params.omega_m)**2.0*(100/c)**4.0
    #print 'cross_const', cross_const
    # 1 square acrminute = sr_const steradian
    sr_const = np.pi**2.0/1.1664e8
    constx = sr_const/cross_const

    # input galaxy number density n(z) file
    idir0 = '/Users/ding/Documents/playground/shear_ps/SVD_ps/'
    inputf = idir0 + 'Input_files/zdistribution_DES_Tully_Fisher.txt'
    lower_z, center_z, upper_z, n_z = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)

    n_sum = np.sum(n_z*(upper_z - lower_z))
    #print(n_sum)
    scale_dndz = 1.0/n_sum                                                  # Normalize n(z) distribution
    n_z = n_z * scale_dndz                                                  # rescale n(z) to match the total number density from the data file equal to 1.0/arcmin^2
    ## spline n_z as a function of z
    tck_nz = interpolate.splrep(center_z, n_z)
    zmax = center_z[-1]
    n_p_zmax = interpolate.splev(zmax, tck_nz, der=1)                       # the first derivative at z_max~1.3

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

    ##--for traditational WL, the total distribution of galaxies n(z) in tomographic bins is unchanged regardless how complicate the photo-z probability distribution is --#
    # calculate the number density nbar^i (integrate dn/dz) in the ith tomographic bin
    # the unit of number density is per steradian
    def n_i_bin(zbin, i):
        zi = zbin[i]
        zf = zbin[i+1]
        n_i = interpolate.splint(zi, zf, tck_nz)
        return n_i

    ##-----set tomographic bins-------------##
    chi_z = np.zeros(num_z)
    for i in range(num_z):
        chi_z[i] = comove_d(center_z[i])*c/100.0
    # we want interpolate z as a function of chi
    tck_zchi = interpolate.splrep(chi_z, center_z)

    tck_chiz= interpolate.splrep(center_z, chi_z)

    #**** Different from the previous case that zmin is 0, zmin is not 0 in the new n(z) file. ****#
    zmin = center_z[0]
    zbin_avg = (zmax-zmin)/float(num_rbin)   # bin interval
    nbin_ext = int(zmax_ext/zbin_avg)
    #print('# of redshift bins (extended): ', nbin_ext)

    # for zbin and chibin, the first element is 0.
    zbin = np.zeros(nbin_ext+1)
    chibin = np.zeros(nbin_ext+1)
    for i in range(nbin_ext+1):
        zbin[i] = i*zbin_avg + zmin
        # Just note that chibin[0] and chibin[1] store the first bin's up boundaries
        chibin[i] = interpolate.splev(zbin[i], tck_chiz, der=0)

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
    z_mid = zmax/2.0
    q_BAO = 110.0   # unit: Mpc/h, the sound horizon scale
    Sigma2_integrand = lambda k: Pk_camb_spl(k) * (1.0 - np.sin(k * q_BAO)/(k * q_BAO))
    pre_factor = 1.0/(3.0 * np.pi**2.0)* (growth_factor(z_mid, cosmic_params.omega_m)/G_0)**2.0
    Sigma2_xy = pre_factor * integrate.quad(Sigma2_integrand, k_camb[0], k_camb[-1], epsabs=1.e-06, epsrel=1.e-06)[0]
    print('At z=', z_mid, 'Sigma2_xy=', Sigma2_xy)

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
    prefix = 'TW_zext_'
    outf_prefix = odir + prefix
    if rank == 0:
        if not os.path.exists(odir):
            os.makedirs(odir)
    comm.Barrier()

    def get_shapenoise():
        shape_noise = np.zeros(nbin_ext)
        # Calculate covariance matrix of Pk, the unit of number density is per steradians
        for i in range(nbin_ext):
            shape_noise[i] = sigmae**2.0/(scale_n * n_i_bin(zbin, i))
        #shape_noise[i] = sigmae**2.0/ s_nz[i] # It's the serious bug that I made and couldn't find it for half a year!
        pseudo_sn = shape_noise*constx

        # put the shape noise (includes the scale factor) in a file
        outf = odir0 + 'mpi_preliminary_data_{}/'.format(Pk_type) + prefix + 'pseudo_shapenoise_{0}rbins_ext.out'.format(nbin_ext)                   # basic variable
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
            #n_l = default_num_l_in_rank * rank + l
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
                    g_i = lens_eff(zbin, center_z, n_z, nz_y2, c_i, z_k, sigma_z_const)
                    if z_k < zmax:  # zmax corresponding to \chi_h in the expression of C^ij(l)
                        GF = (growth_factor(z_k, cosmic_params.omega_m)/G_0)**2.0
                        #print('zmax, z_k, GF:', zmax, z_k, GF)
                        c_j = c_i
                        # here g_row is the row index of gmatrix_jk
                        for g_row in range(g_nr):
                            g_j = lens_eff(zbin, center_z, n_z, nz_y2, c_j, z_k, sigma_z_const)
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
            #n_l = default_num_l_in_rank * rank + l
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
                        gi = lens_eff(zbin, center_z, n_z, nz_y2, rb_i, z_k, sigma_z_const)
                        # redshift bin j: rb_j
                        rb_j = iu1[1][i]
                        gj = lens_eff(zbin, center_z, n_z, nz_y2, rb_j, z_k, sigma_z_const)
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
        photoz_matrix = np.array([], dtype='float64').reshape(0, num_z) # we may need more data points to make curv smooth. --05/24/2018
        for bin_id in range(nbin_ext):
            photoz_array = np.zeros(num_z)
            i = 0
            for z_sp in center_z:
                ni_zp = photo_ni(zbin, center_z, n_z, nz_y2, bin_id, z_sp, sigma_z_const)  # function photo_ni function outputs n_i(z) at redshift z
                # Since n_z has been normalized, we need to take account of scale_n
                photoz_array[i] = ni_zp * scale_n
                i = i+1
            photoz_matrix = np.vstack([photoz_matrix, photoz_array])

        ofile = odir + "gal_numden_photoz_{}rbins.out".format(nbin_ext)
        header_line = " z(spectroscopic)   n_i(z) for each ith photo-z bin"  # first column is z, the other columns are n_i(z) with increasing bin id
        np.savetxt(ofile, np.vstack([center_z, photoz_matrix]).T, fmt='%.7e', header=header_line, newline='\n', comments='#')

    def plot_numd_photoz(n_z, tck_nz):
        idir = "./numd_distribute_photoz/"
        odir = idir + "nz_fig/"
        if not os.path.exists(odir):
            os.makedirs(odir)

        ifile = idir + "gal_numden_photoz_{}rbins.out".format(nbin_ext)
        #-- simply show the results ---#
        data_m = np.loadtxt(ifile, dtype='f8', delimiter=' ', comments='#')
        center_z = data_m[:, 0]
        photoz_matrix = data_m[:, 1:]
        n_z = n_z * scale_n     # since n_z has been normalized, we need to add back scale_n to obtain the true n_z value
        #print(photoz_matrix)
        fig, ax = plt.subplots(figsize=(8, 6))
        for bin_id in range(nbin_ext):
            if bin_id == 11:   # Show the 12th tomographic bin, which has the largest n^i(z)
                ax.plot(center_z, photoz_matrix[:, bin_id], 'r-.', lw=1.0)
            else:
                ax.plot(center_z, photoz_matrix[:, bin_id], 'k-.', lw=0.5)
        #ax.axvline(zbin[0], color='grey', linestyle=':')
        ax.plot(center_z, n_z, 'k-', lw=2.0)
        ax.set_xlabel(r'$z$', fontsize=20)
        ax.set_xlim([0.0, zmax])
        ymax = 17        # y range maximum
        ax.set_ylim([0.0, ymax])
        for bin_id in range(num_rbin):
            ax.axvline(zbin[bin_id], ymin=0.0, ymax=interpolate.splev(zbin[bin_id], tck_nz, der=0)*scale_n/ymax, color='grey', linestyle=':')

        ax.minorticks_on()
        ax.tick_params('both', length=5, width=2, which='major', labelsize=15)
        ax.tick_params('both', length=3, width=1, which='minor')

        ax.set_ylabel(r'$dn^i/dz \; [\mathrm{arcmin}]^{-2}$', fontsize=20)
        ax.set_title("PWL-Stage III", fontsize=20)
        plt.tight_layout()
        figname = "gal_numden_{}rbins_photoz.pdf".format(num_rbin)
        plt.savefig(odir + figname)
        plt.show()

    if show_nz == "True" and rank == 0:
        get_photoz_density(center_z)
        plot_numd_photoz(n_z, tck_nz)


if __name__ == '__main__':
    main()

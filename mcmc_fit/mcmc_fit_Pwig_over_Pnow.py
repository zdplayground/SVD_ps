#!/Users/ding/anaconda3/bin/python
# 1. Copy the code from mcmc_fit_nonlinear_BAO.py. Modify it to fit Pwig devided by Pnow. Pwig and Pnow are both extracted from SVD. Still use covariance matrix of P(k).
# -- 09/17/2017
# 2. Add the option whether we apply SV threshold (SVc) on extracting the inverse covariance matrix of Pwnw. --05/04/2018
# 3. Add the optional parameter Psm_type (either Pnorm or Pnow). -- 05/21/2018
# 4. Add the argument "start_lmin" to skip small angular modes of C^ij(l) and G matrix due to f_sky effect. --06/06/2018
# 5. Find the bug of Sigma2_xy fixed for both stage III and IV surveys. Sigma2_xy should be calculated at z=1.0 for stage IV. --06/22/2018
# 6. Add the argument "alpha" to fix the BAO scale shifting parameter. Modify the output result with fixed parameter value.
#    Add 'if conditon' that Sigma2_xy=-np.inf if params_str=001. This setting is to calcuate the significance of BAO detection. --06/30/2018
# 7. Add a new argument "Sigma2_inf" to distinguish the value of Sigma2_xy when we set it as a fixed parameter. --07/05/2018
#
#--------------------------------------
# Similar as the mcmc fitting code in BAO systematics project, we use the routine to fit spatial power spectrum extracted from lesing shear power spectrum. -- 08/10/2017
# Since the power spectrum is for matter, there is no galaxy bias or RSD issues. Hence, the model could be much simpler than the galaxy power spectrum in redshift space.
# We use the fitting model as Pwig(k')/Pnow(k')=1+(Pwig(k)/Pnow(k)-1)*exp(-k^2*Sigma^2/2).
# Add parameter A as the amplitude parameter. -- 08/17/2017
# Add the condition of Pk type. If for no-wiggle, we set parameter Sigma as a large number, but keep the model to be the same. -- 08/23/2017
#---------------------------------------
# 09/07/2017, marginalize Fisher matrix of P(k) for fitting in some certain k range.
# 09/14/2017, set Sigma from theoretical value, do fitting with fixed Sigma.
# 09/16/2017, after correcting the fitting model (take parameter A as the scale of power spectrum amplitude), we repeat the previous fitting.
# --------------------------------------
# Add judgment of mpi_used. See whether MPI used or not to calculate Cij_l and Gm_prime. -- 10/11/2017
# Add f_sky parameter to include fitting data files in the folder TF_cross-ps. Not include f_sky in other cases (may need to polish the code). --11/01/2017
import emcee
from emcee.utils import MPIPool
from mpi4py import MPI
import time
import numpy as np
import scipy.optimize as op
from scipy import linalg
from scipy.interpolate import InterpolatedUnivariateSpline
from functools import reduce
import os, sys
from lnprob_nonlinear import match_params, cal_pk_model, lnprior
from mcmc_funs import gelman_rubin_convergence, write_params, set_params
import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
import argparse
sys.path.append("../")
import cosmic_params  # module contains cosmological parameters


# Input observed Pk_obs(k') = Pwig(k')/Pnow(k').
# If the observed and input power spectrum have different redshifts, we need norm_gf as the normalized growth factor D(z).
# Change variable ivar to icov.
def lnlike(theta, params_indices, fix_params, k_p, Pk_obs, icov, Pwig_spl, Psm_spl, norm_gf):
    alpha, Sigma2_xy, A = match_params(theta, params_indices, fix_params)

    coeff = 1.0/alpha
    k_t = k_p*coeff        # k=k'/alpha
    Pk_linw = Pwig_spl(k_t)
    Pk_sm = Psm_spl(k_t)
    Pk_model = cal_pk_model(Pk_linw, Pk_sm, k_t, Sigma2_xy, A)
    diff = Pk_model - Pk_obs
    return -0.5* reduce(np.dot, (diff, icov, diff))

def lnprob(theta, params_indices, fix_params, k_p, Pk_obs, icov, Pwig_spl, Psm_spl, norm_gf):
    lp = lnprior(theta, params_indices, fix_params)
    if (lp < -1.e20):
        return -np.inf
    return lp + lnlike(theta, params_indices, fix_params, k_p, Pk_obs, icov, Pwig_spl, Psm_spl, norm_gf)

# Find the maximum likelihood value.
chi2 = lambda *args: -2 * lnlike(*args)


# MCMC routine
def mcmc_routine(ndim, N_walkers, theta, params_T, params_indices, fix_params, k_range, Pk_wnow_obs, icov_Pk_wnow, Pwig_spl, Psm_spl, norm_gf, params_name, pool):
    ti = time.time()

    Nchains = 4
    minlength = 800
    epsilon = 0.01
    ichaincheck = 50
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    result = op.minimize(chi2, theta, args=(params_indices, fix_params, k_range, Pk_wnow_obs, icov_Pk_wnow, Pwig_spl, Psm_spl, norm_gf), method='Powell')
    theta_optimize = result["x"]
    print("Parameters from Powell optimization: ", theta_optimize) # only output parameters which are free to change

    theta_optimize = theta
    print("Initial parameters for MCMC: ", theta_optimize)

    pos = []
    sampler = []
    rstate = np.random.get_state()
    # Set up the sampler.
    for jj in range(Nchains):
        pos.append([theta_optimize + params_T*np.random.uniform(-1.0, 1.0, ndim) for i in range(N_walkers)])

        sampler.append(emcee.EnsembleSampler(N_walkers, ndim, lnprob, a=2.0, args=(params_indices, fix_params, k_range, Pk_wnow_obs, icov_Pk_wnow, Pwig_spl, Psm_spl, norm_gf), pool=pool))
    print(type(sampler))

    # Clear and run the production chain.
    print("Running MCMC...")

    withinchainvar = np.zeros((Nchains,ndim))
    meanchain = np.zeros((Nchains,ndim))
    scalereduction = np.arange(ndim,dtype=np.float)
    for jj in range(0, ndim):
        scalereduction[jj] = 2.

    itercounter = 0
    chainstep = minlength
    loopcriteria = 1
    num_iteration = 1
    while loopcriteria and num_iteration<50:
        itercounter = itercounter + chainstep
        print("chain length =",itercounter," minlength =",minlength)

        for jj in range(Nchains):
            # Since we write the chain to a file we could put storechain=False, but in that case
            # the function sampler.get_autocorr_time() below will give an error
            for result in sampler[jj].sample(pos[jj], iterations=chainstep, rstate0=np.random.get_state(), storechain=True, thin=1):
                pos[jj] = result[0]
                #print(pos)
                chainchi2 = -2.*result[1]
                rstate = result[2]

            # we do the convergence test on the second half of the current chain (itercounter/2)
            chainsamples = sampler[jj].chain[:, itercounter//2:, :].reshape((-1, ndim))
            #print("len chain = ", chainsamples.shape)
            withinchainvar[jj] = np.var(chainsamples, axis=0)
            meanchain[jj] = np.mean(chainsamples, axis=0)

        scalereduction = gelman_rubin_convergence(withinchainvar, meanchain, itercounter//2, Nchains, ndim)
        print("scalereduction = ", scalereduction)

        loopcriteria = 0
        for jj in range(0, ndim):
            if np.absolute(1.0-scalereduction[jj]) > epsilon:
                loopcriteria = 1

        chainstep = ichaincheck
        num_iteration = num_iteration + 1
    print("Done.")

    # Print out the mean acceptance fraction. In general, acceptance_fraction
    # has an entry for each walker so, in this case, it is a 250-dimensional vector.
    for jj in range(0, Nchains):
        print("Mean acceptance fraction for chain ", jj,": ", np.mean(sampler[jj].acceptance_fraction))
    # Estimate the integrated autocorrelation time for the time series in each parameter.
    #for jj in range(0, Nchains):
    #    print("Autocorrelation time for chain ", jj,": ", sampler[jj].get_autocorr_time())
    ###################################
    ## Compute the quantiles ##########
    ###################################

    #samples=[]
    mergedsamples=[]

    for jj in range(0, Nchains):
        #samples.append(sampler[jj].chain[:, itercounter/2:, :].reshape((-1, ndim)))
        mergedsamples.extend(sampler[jj].chain[:, itercounter//2:, :].reshape((-1, ndim)))
    print("length of merged chain = ", sum(map(len,mergedsamples))//ndim)

    theta_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(mergedsamples, [15.86555, 50, 84.13445], axis=0)))
    theta_mcmc = list(theta_mcmc)

    print("MCMC result: ")
    for i in range(len(theta)):
        print("{0}={1[0]}+{1[1]}-{1[2]}".format(params_name[i], theta_mcmc[i]))

    del sampler
    tf = time.time()
    print("One mcmc running set time: ", tf-ti)
    return np.array(theta_mcmc)

def filter_krange(k_all, kmin, kmax):
    indices = []
    for i in range(len(k_all)):
        if k_all[i] >= kmin and k_all[i] <= kmax:
            indices.append(i)
    return indices

def filter_invCov_on_k(ifile_Cov, k_indices):
    npzfile = np.load(ifile_Cov)
    icov_Pk_wnw = npzfile['arr_0']
    #print('icov_Pk_wnw: ', icov_Pk_wnw)
    cov_Pwnw = linalg.inv(icov_Pk_wnw)
    #print('cov_Pwnw: ', cov_Pwnw)
    identity = np.dot(cov_Pwnw, icov_Pk_wnw)
    #print('cov * icov: ', identity)
    #print('Inverse process is good?', np.allclose(identity, np.eye(num_kbin)))  # test whether the inverse process is very successful
    part_cov_Pwnw = cov_Pwnw[np.ix_(k_indices, k_indices)]
    part_icov_Pk_wnw = linalg.inv(part_cov_Pwnw)
    npzfile.close()
    return part_icov_Pk_wnw

# Fit extracted power spectrum from shear power spectrum. We use Pwig/Pnow as the observable.
def fit_BAO():
    parser = argparse.ArgumentParser(description='Use mcmc routine to get the BAO peak stretching parameter alpha and damping parameter, made by Zhejie.',\
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lt", help = '*The type of weak lensing survey. (TF: Tully-Fisher; TW: traditional (photo-z) weak lensing.)', required=True)
    parser.add_argument("--nrbin", help = '*Number of tomographic bins.', type=int, required=True)
    parser.add_argument("--nkbin", help = '*Number of output k bins.', type=int, required=True)
    parser.add_argument("--shapenf", help = '*Shape noise factor.', required=True)
    parser.add_argument("--kmin", help = '*kmin fit boundary.', required=True)
    parser.add_argument("--kmax", help = '*kmax fit boundary.', required=True)
    parser.add_argument("--params_str", help = 'Set fitting parameters. 1: free; 0: fixed.', required=True)
    parser.add_argument("--Sigma2_inf", help = 'Whether setting Sigma2_xy as infinity or not.', default='False')
    parser.add_argument("--alpha", help = 'Fix the parameter alpha value.', default=1.0, type=np.float)
    parser.add_argument("--Pwig_type", help = '*The spatial P(k)_wig whether is linear or not. Type Pwig_linear or Pwig_nonlinear; in nonlinear, BAO is damped.', required=True)
    parser.add_argument("--Psm_type", help = 'The expression of Pnorm. The default case, Pnorm from Eisenstein & Zaldarriaga 1999. \
                                              Test Pnorm=Pnow, which is derived from transfer function.')
    #parser.add_argument("--Sigma", help = 'BAO damping parameter Sigma value. (Either 0.0 or 100.0.)', type=float, required=True)
    parser.add_argument("--survey_stage", help = 'Optional parameter. KW_stage_IV (kinematic weak lensing) or PW_stage_IV (photo-z). It could also be considered as the directory of data\
                        files to be fitted.')
    #parser.add_argument("--mpi_used", help = 'Whether MPI is implemented in the calculation of Cijl_prime, Gm_prime. Either True or False.')
    parser.add_argument("--f_sky", help = 'This addtional argument is for data files in TF_cross-ps. Distinguish cases with different f_sky value.')
    parser.add_argument("--set_SVc_on_CovP", help = '*Whether we replace smaller SV to be SVc in W matrix for the output inverse covariance matrix of Pk. Either True or False.', required=True)
    parser.add_argument("--start_lmin", help = "*The minimum ell value considered in the analysis. Default is 1. For Stage III, it's 10 for Stage III and 4 for Stage IV", default=1, type=int)

    ##parser.add_argument("--num_eigv", help = 'Number of eigenvalues used from SVD.', required=True)
    args = parser.parse_args()
    #print("args: ", args.lt)
    lt = args.lt
    num_rbin = args.nrbin
    num_kbin = args.nkbin
    shapenf = args.shapenf
    kmin = float(args.kmin)
    kmax = float(args.kmax)
    params_str = args.params_str
    params_indices = [int(i) for i in params_str]
    Pwig_type = args.Pwig_type
    Psm_type = args.Psm_type
    survey_stage = args.survey_stage
    #mpi_used = args.mpi_used
    f_sky = args.f_sky
    set_SVc_on_CovP = args.set_SVc_on_CovP
    start_lmin = args.start_lmin

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    lt_prefix = {'TF': 'TF', 'TW': 'TW_zext'}
    old_stdout = sys.stdout
    if survey_stage:
        odir = './{}/fit_kmin{}_kmax{}_Pwig_over_Pnow/{}/'.format(survey_stage, kmin, kmax, Pwig_type)
    else:
        odir = './fit_kmin{}_kmax{}_Pwig_over_Pnow_fsky{}/{}/'.format(kmin, kmax, f_sky, Pwig_type)
    # if mpi_used == 'True':
    #     odir = './{}/fit_kmin{}_kmax{}_Pwig_over_Pnow/mpi_{}/'.format(survey_stage, kmin, kmax, Pwig_type)

    if Psm_type == 'Pnow':
        odir = odir + 'set_Pnorm_Pnow/'
    if start_lmin != 1:
        odir = odir + 'start_ell_{}/'.format(start_lmin)

    if rank == 0:
        if not os.path.exists(odir):
            os.makedirs(odir)
    comm.Barrier()

    Sigma2_xy_dict = {'stage_III': 31.176, 'stage_IV': 22.578}
    for ss_name in Sigma2_xy_dict:
        if ss_name in survey_stage:
            stage_name = ss_name
    if Pwig_type == 'Pwig_nonlinear':
        Sigma2_xy = Sigma2_xy_dict[stage_name]
    elif Pwig_type == 'Pwig_linear':
        Sigma2_xy = 0.0

    if args.Sigma2_inf == 'True':
        Sigma2_xy = np.inf

    ofile = odir + "mcmc_fit_{}_{}rbin_{}kbin_snf{}_params{}_Sigma2_{}.log".format(lt, num_rbin, num_kbin, shapenf, params_str, Sigma2_xy)
    if params_str == '001':
        ofile = odir + "mcmc_fit_{0}_{1}rbin_{2}kbin_snf{3}_params{4}_alpha{5:.2f}_Sigma2_{6}.log".format(lt, num_rbin, num_kbin, shapenf, params_str, args.alpha, Sigma2_xy)
    log_file = open(ofile, "w")
    sys.stdout = log_file
    print('Arguments of fitting: ', args)

    ifile = '../Input_files/CAMB_Planck2015_matterpower.dat'
    kcamb, Pkcamb = np.loadtxt(ifile, dtype='f8', comments='#', unpack=True)
    Pwig_spl = InterpolatedUnivariateSpline(kcamb, Pkcamb)
    k_0 = 0.001         # unit h*Mpc^-1
    Pk_0 = Pwig_spl(k_0)

    ifile = '../Input_files/transfer_fun_Planck2015.dat'
    kk, Tf = np.loadtxt(ifile, dtype='f8', comments='#', usecols=(0,1), unpack=True)
    #print kk==kcamb
    Tf_spl = InterpolatedUnivariateSpline(kk, Tf)
    Tf_0 = Tf_spl(k_0)

    P0_a = Pk_0/(pow(k_0, cosmic_params.ns) * Tf_0**2.0)
    Psm = P0_a * pow(kcamb, cosmic_params.ns) * Tf**2.0               # Get primordial (smooth) power spectrum from the transfer function
    Psm_spl = InterpolatedUnivariateSpline(kcamb, Psm)

    norm_gf = 1.0
    N_walkers = 40

    all_param_names = 'alpha', 'Sigma2_xy', 'A'
    all_temperature = 0.01, 1.0, 0.1

    alpha, A = args.alpha, 1.0      # initial guess for fitting, the value of Sigma2_xy at z=0.65 is from theory prediction (see code TF_cross_convergence_ps_bin.py)
    all_params = alpha, Sigma2_xy, A
    N_params, theta, fix_params, params_T, params_name = set_params(all_params, params_indices, all_param_names, all_temperature)

    # Fit for DM power spectrum
    if lt == 'TF':
        idir0 = '../{}_cross-ps/'.format(lt)
        if survey_stage:
            idir0 = '../{}/'.format(survey_stage)
        idir1 = '{}_Pk_output_dset_{}/'
        # if mpi_used == 'True':
        #     idir1 = 'mpi_{}_Pk_output_dset_{}/'
        if f_sky:
            idir1 = '{}_Pk_output_dset_{}_fsky{}/'
    elif lt == 'TW':
        idir0 = '../{}_f2py_SVD/'.format(lt)
        if survey_stage:
            idir0 = '../{}/'.format(survey_stage)
        idir1 = '{}_Pk_output_dset_{}/'

    idir2 = '{}rbins_{}kbins_snf{}/'.format(num_rbin, num_kbin, shapenf)

    odir1 = 'mcmc_fit_params_Pwig_over_Pnow/{}/kmin{}_kmax{}/{}rbins_{}kbins_snf{}/'.format(Pwig_type, kmin, kmax, num_rbin, num_kbin, shapenf)
    if f_sky:
        odir1 = 'mcmc_fit_params_Pwig_over_Pnow/{}/kmin{}_kmax{}/{}rbins_{}kbins_snf{}_fsky{}/'.format(Pwig_type, kmin, kmax, num_rbin, num_kbin, shapenf, f_sky)
    # if mpi_used == 'True':
    #     odir1 = 'mcmc_fit_params_Pwig_over_Pnow/{}/kmin{}_kmax{}/mpi_{}rbins_{}kbins_snf{}/'.format(Pwig_type, kmin, kmax, num_rbin, num_kbin, shapenf)
    if Psm_type == 'Pnow':
        idir2 = idir2 + 'set_Pnorm_Pnow/'
        odir1 = odir1 + 'set_Pnorm_Pnow/'

    if start_lmin != 1 :
        idir2 = idir2 + 'start_ell_{}/'.format(start_lmin)
        odir1 = odir1 + 'start_ell_{}/'.format(start_lmin)

    if params_str == '001':
        odir1 = odir1 + 'alpha_{0:.2f}/'.format(args.alpha)

    if f_sky:
        idir = idir0 + idir1.format(lt_prefix[lt], Pwig_type, f_sky) + idir2  # not exactly matching format, but it's ok if there is no f_sky parameter.
    else:
        idir = idir0 + idir1.format(lt_prefix[lt], Pwig_type) + idir2
    odir = idir0 + odir1
    if rank == 0:
        if not os.path.exists(odir):
            os.makedirs(odir)

    ifile = idir + 'Pk_wnw_{}rbin_{}kbin_withshapenoisefactor{}_{}eigenvW.out'.format(num_rbin, num_kbin, shapenf, num_kbin)
    k_all = np.loadtxt(ifile, dtype='f8', comments='#', usecols=(0,))
    indices = filter_krange(k_all, kmin, kmax)  # Here we don't need to use sigma_Pk_wnw anymore.
    N_fitbin = len(indices)
    print('# of fit k bins: ', N_fitbin)
    print('The indices of fitting k bins: ', indices)

    if set_SVc_on_CovP != 'True':
        ifile = idir + 'Cov_Pwnw_inv_{}rbin_{}kbin_withshapenoisefactor{}.npz'.format(num_rbin, num_kbin, shapenf)
        npzfile = np.load(ifile)
        icov_Pk_wnw = npzfile['arr_0']
        print('icov_Pk_wnw: ', icov_Pk_wnw)

        cov_Pwnw = linalg.inv(icov_Pk_wnw)
        print('cov_Pwnw: ', cov_Pwnw)
        identity = np.dot(cov_Pwnw, icov_Pk_wnw)
        print('cov * icov: ', identity)
        print('Inverse process is good?', np.allclose(identity, np.eye(num_kbin)))  # test whether the inverse process is very successful

        part_cov_Pwnw = cov_Pwnw[np.ix_(indices, indices)]
        part_icov_Pk_wnw = linalg.inv(part_cov_Pwnw)
        print('Inverse process for marginalized matrix is good?', np.allclose(np.dot(part_cov_Pwnw, part_icov_Pk_wnw), np.eye(N_fitbin)))
        print('The inverse covariance matrix for fitting: ', part_icov_Pk_wnw)

    pool = MPIPool(loadbalance=True)
    #for num_eigv in range(num_kbin//3, num_kbin+1):
    for num_eigv in range(1, num_kbin+1):
        np.random.seed(1)
        if f_sky:
            idir_Pwig = idir0 + idir1.format(lt_prefix[lt], Pwig_type, f_sky) + idir2
            idir_Pnow = idir0 + idir1.format(lt_prefix[lt], 'Pnow', f_sky) + idir2
        else:
            idir_Pwig = idir0 + idir1.format(lt_prefix[lt], Pwig_type) + idir2
            idir_Pnow = idir0 + idir1.format(lt_prefix[lt], 'Pnow') + idir2
        ifile = idir_Pwig + 'Pk_wig_{}rbin_{}kbin_withshapenoisefactor{}_{}eigenvW.out'.format(num_rbin, num_kbin, shapenf, num_eigv)
        print(ifile)
        data_m = np.loadtxt(ifile, dtype='f8', comments='#') # k, P(k), sigma_Pk
        k_obs, Pk_wig_obs = data_m[indices, 0], data_m[indices, 1]

        ifile = idir_Pnow + 'Pk_now_{}rbin_{}kbin_withshapenoisefactor{}_{}eigenvW.out'.format(num_rbin, num_kbin, shapenf, num_eigv)
        print(ifile)
        data_m = np.loadtxt(ifile, dtype='f8', comments='#')
        k_obs, Pk_now_obs = data_m[indices, 0], data_m[indices, 1]

        Pk_wnw_obs = Pk_wig_obs/Pk_now_obs

        if set_SVc_on_CovP == 'True':
            ifile = idir_Pwig + 'Cov_Pwnw_inv_{}rbin_{}kbin_withshapenoisefactor{}_{}eigenvW_SVc.npz'.format(num_rbin, num_kbin, shapenf, num_eigv)
            part_icov_Pk_wnw = filter_invCov_on_k(ifile, indices)

        params_mcmc = mcmc_routine(N_params, N_walkers, theta, params_T, params_indices, fix_params, k_obs, Pk_wnw_obs, part_icov_Pk_wnw, Pwig_spl, Psm_spl, norm_gf, params_name, pool)
        print(params_mcmc)
        chi_square = chi2(params_mcmc[:, 0], params_indices, fix_params, k_obs, Pk_wnw_obs, part_icov_Pk_wnw, Pwig_spl, Psm_spl, norm_gf)
        reduced_chi2 = chi_square/(N_fitbin-N_params)
        print("chi^2/dof: ", reduced_chi2, "\n")
        if params_str == '001':
            # in order to distinguish the Sigma2_xy value, we need to specify it.
            filename = 'Pk_wnw_{}rbin_{}kbin_snf{}_{}eigenvW_params{}_Sigma2_{}.dat'.format(num_rbin, num_kbin, shapenf, num_eigv, params_str, Sigma2_xy)
        else:
            filename = 'Pk_wnw_{}rbin_{}kbin_snf{}_{}eigenvW_params{}.dat'.format(num_rbin, num_kbin, shapenf, num_eigv, params_str) # Sigma2_xy is fixed in the default case
        ofile_params = odir + filename
        print('ofile_params:', ofile_params)
        write_params(ofile_params, params_mcmc, params_name, reduced_chi2, fix_params)

    pool.close()

    sys.stdout = old_stdout
    log_file.close()

def main():
    fit_BAO()

if __name__ == '__main__':
    main()

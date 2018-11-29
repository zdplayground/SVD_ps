#!/Users/ding/anaconda3/bin/python
# -*- coding: utf-8 -*-
# Show lens efficiency g^i(chi) for PW Stage-IV. -- 03/23/2018
# 1. Modify z_array, extend the maximum of z_array to zmax. --07/09/2018
# 2. Normalize the n(z) distribution. --08/31/2018
#
import numpy as np
import math
import os, sys
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/')
import cosmic_params
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import integrate
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/PW_modules/')
from lens_eff_module import lens_eff
sys.path.append('/Users/ding/Documents/playground/shear_ps/SVD_ps/common_modules/')
from module_market import dis_fun, comove_d
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Use mcmc routine to get the BAO peak stretching parameter alpha and damping parameter, made by Zhejie.')
parser.add_argument("--survey_stage", help = 'Survey stage', default='PW_stage_IV')
parser.add_argument("--nrbin", help = '*Number of tomographic bins.', type=int, required=True)
parser.add_argument("--z_target", help = '*The target z value where the lens efficiency is calculated.', type=float, required=True)

##parser.add_argument("--num_eigv", help = 'Number of eigenvalues included from SVD.', required=True)
args = parser.parse_args()
survey_stage = args.survey_stage
nrbin = args.nrbin                 # for photo-z<1.3
z_target = args.z_target


def plot_lens_eff(z_array, g_i_array, nrbin, z_target, figname):
    fig, ax = plt.subplots()
    ax.plot(z_array, g_i_array)
    #ax.set_ylim([-0.002, 0.05])
    ax.grid('on')
    textline = 'PW Stage-IV\n' + r'$N_{\mathrm{zbin}}$' + '={}\n'.format(nrbin) + r'$z_k={}$'.format(z_target)
    ax.text(z_target-0.2, 0.035, textline, fontsize=15)
    ax.set_xlabel(r'$z$', fontsize=20)
    ax.set_ylabel(r'$g^i(z)$', fontsize=20)
    ax.tick_params('both', length=5, width=2, which='major', labelsize=15)
    fig.tight_layout()
    plt.savefig(figname)
    #plt.show()
    plt.close()

# On top of the lens_eff, we add factor \chi/a(z) in front.
def plot_lens_eff_version2(z_array, chi_array, g_i_array, nrbin, z_target, figname):
    fig, ax = plt.subplots()
    y = chi_array * (1.0 + z_array) * g_i_array
    ax.plot(z_array, y)
    #ax.set_ylim([-0.0005, 0.012])
    ax.grid('on')
    textline = 'PW Stage-IV\n' + r'$N_{\mathrm{zbin}}$' + '={}\n'.format(nrbin) + r'$z_k={}$'.format(z_target)
    ax.text(z_target-0.2, 0.008, textline, fontsize=15)
    ax.set_xlabel(r'$z$', fontsize=20)
    #ax.set_ylabel(r'$\frac{3 H_0^2 \, \chi \Omega_m}{2 c^2\, a(z)} g^i(z)$', fontsize=18)
    ax.set_ylabel(r'$\frac{\chi}{a(z)} g^i(z)$', fontsize=18)
    ax.tick_params('both', length=5, width=2, which='major', labelsize=15)
    fig.tight_layout()
    plt.savefig(figname)
    #plt.show()
    plt.close()

# On top of the lens_eff, we add factor 1/a(z) in front.
def plot_lens_eff_version3(z_array, g_i_array, nrbin, z_target, figname):
    fig, ax = plt.subplots()
    y = (1.0 + z_array) * g_i_array
    ax.plot(z_array, y)
    #ax.set_ylim([-0.0005, 0.06])
    ax.grid('on')
    textline = 'PW Stage-IV\n' + r'$N_{\mathrm{zbin}}$' + '={}\n'.format(nrbin) + r'$z_k={}$'.format(z_target)
    ax.text(z_target-0.2, 0.008, textline, fontsize=15)
    ax.set_xlabel(r'$z$', fontsize=20)
    #ax.set_ylabel(r'$\frac{3 H_0^2 \, \chi \Omega_m}{2 c^2\, a(z)} g^i(z)$', fontsize=18)
    ax.set_ylabel(r'$\frac{1}{a(z)} g^i(z)$', fontsize=18)
    ax.tick_params('both', length=5, width=2, which='major', labelsize=15)
    fig.tight_layout()
    plt.savefig(figname)
    #plt.show()
    plt.close()

def main():
    c = 2.99792458e5      # speed of light unit in km/s
    sigma_z_const = 0.05  # for PW-Stage IV, from Table 3 in Eric et al. 2015
    #scale_n = 31.0
    # input galaxy number density n(z) file
    inputf = '../Input_files/nz_stage_IV.txt'             # Input file of n(z) which is the galaxy number density distribution in terms of z
    # Here center_z denotes z axis of n(z). It may not be appropriate since we don't have redshift bin setting
    center_z, n_z = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
    num_z = len(center_z)
    spl_nz = InterpolatedUnivariateSpline(center_z, n_z)
    n_sum = spl_nz.integral(center_z[0], center_z[-1])                      # Calculate the total number density
    #print(n_sum)
    scale_dndz = 1.0/n_sum
    n_z = n_z * scale_dndz                                                  # normalize n(z) to match the total number density from the data file equal to 1.0/arcmin^2
    tck_nz = interpolate.splrep(center_z, n_z)
    zmax = 2.0                             # Based on Eric's n(z) data file for Stage IV weak lensing survey, we cut number density at z=2.0, after which n(z) is very small.
    #----------------------------------------------------------------------------------------------
    #-- Removed this extrapolation part for n(z) since the given data file covers large z range.
    # ...
    #----------------------------------------------------------------------------------------------
    # Set zmax_ext as the maximum z after extension, it's about 2.467 in this case.
    zmax_ext = zmax + math.sqrt(2.0)*0.05*(1+zmax)*2.2   # 2.2 is from erf(2.2) which is close to 1.0
    print('zmax_ext: ', zmax_ext)   # It's about 2.467.
    nz_y2 = np.zeros(num_z)

    #**** Different from the previous case that zmin is 0, zmin is not 0 in the new n(z) file. ****#
    zmin = center_z[0]
    zbin_avg = (zmax-zmin)/float(nrbin)   # bin interval
    nbin_ext = int(zmax_ext/zbin_avg)
    #print('# of redshift bins (extended): ', nbin_ext)

    zbin = np.zeros(nbin_ext+1)
    for i in range(nbin_ext+1):
        zbin[i]=i*zbin_avg + zmin
        if zbin[i] <= z_target:
            target_i = i

    #z_array = np.linspace(0.0, zbin[target_i+1], 1000, endpoint=False)
    z_array = np.linspace(0.0, zmax, 1000, endpoint=False)
    chi_array = np.array([comove_d(z_i) for z_i in z_array])   # not including the unit of distance
    g_i_array = np.array([], dtype=np.float64)
    for z_k in z_array:
        g_i = lens_eff(zbin, center_z, n_z, nz_y2, target_i, z_k, sigma_z_const)
        g_i_array = np.append(g_i_array, g_i)

    odir0 = '/Users/ding/Documents/playground/shear_ps/project_final/fig_lens_eff/gi_data/'
    odir = odir0 + '{}/'.format(survey_stage)
    if not os.path.exists(odir):
        os.makedirs(odir)
    ofile = odir + 'gi_nrbin{}_zk_{}_rbinid_{}.npz'.format(nrbin, z_target, target_i)
    #ofile = './gi_nrbin{}_zk_{}_rbinid_{}_30abscissas.npz'.format(nrbin, z_target, target_i)
    #ofile = './gi_nrbin{}_zk_{}_rbinid_{}_50abscissas.npz'.format(nrbin, z_target, target_i)
    #ofile = './gi_nrbin{}_zk_{}_rbinid_{}_50abscissas_5sigma.npz'.format(nrbin, z_target, target_i)
    #ofile = './gi_nrbin{}_zk_{}_rbinid_{}_50abscissas_largerintlim_in_denominator.npz'.format(nrbin, z_target, target_i)
    np.savez(ofile, z=z_array, gi=g_i_array)

    odir = './figs/lens_eff_gi/'
    if not os.path.exists(odir):
        os.makedirs(odir)

    ofile = odir + 'gi_nrbin{}_zk_{}.pdf'.format(nrbin, z_target)
    plot_lens_eff(z_array, g_i_array, nrbin, z_target, ofile)

    ofile = odir + 'gi_nrbin{}_zk_{}_version2.pdf'.format(nrbin, z_target)
    plot_lens_eff_version2(z_array, chi_array, g_i_array, nrbin, z_target, ofile)

    #ofile = odir + 'gi_nrbin{}_zk_{}_version3.pdf'.format(nrbin, z_target)
    #plot_lens_eff_version3(z_array, g_i_array, nrbin, z_target, ofile)


if __name__ == '__main__':
    main()

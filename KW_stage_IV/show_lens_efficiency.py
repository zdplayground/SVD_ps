#!/Users/ding/miniconda3/bin/python
# -*- coding: utf-8 -*-
# Show lens efficiency g^i(chi) for KW Stage-IV. -- 03/23/2018
# We ignore the unit of comoving distanceself.
#
import numpy as np
import os, sys
sys.path.append("../")
import cosmic_params
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import integrate
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Simulate lensing convergence power spectrum C^ij(l), made by Zhejie.')
parser.add_argument("--survey_stage", help = '*Survey stage', required=True)
parser.add_argument("--nrbin", help = '*Number of tomographic bins.', type=int, required=True)
parser.add_argument("--z_target", help = '*The target z value where the lens efficiency is calculated.', type=float, required=True)

args = parser.parse_args()
survey_stage = args.survey_stage
nrbin = args.nrbin                      # nrbin represents the number of tomographic bins
z_target = args.z_target

# It's from expression 1/H(z) without the constant
def dis_fun(z):
    return 1.0/np.sqrt(cosmic_params.omega_m*(1.0+z)**3.0+cosmic_params.omega_L)

# Calculate the comoving distance without c/H_0 constant
def comove_d(z):
    distance = integrate.quad(dis_fun, 0.0, z)
    return distance[0]

# Define a function calculating lens efficiency.
def lens_eff(chibin, i, chi_k):
    x_low = max(chibin[i], chi_k)
    x_up = chibin[i+1]
    g_i = 1.0 - (np.log(x_up)-np.log(x_low))/(x_up-x_low) *chi_k             # The result g_i is obtained from theoretical integration.
    return g_i

def plot_lens_eff(z_array, g_i_array, nrbin, z_target, figname):
    fig, ax = plt.subplots()
    ax.plot(z_array, g_i_array)
    ax.grid('on')
    textline = 'KW Stage-IV\n'+'nrbin={}\n'.format(nrbin) + r'$z_k={}$'.format(z_target)
    ax.text(z_target-0.2, 0.8, textline, fontsize=12)
    ax.set_xlabel(r'$z$', fontsize=20)
    ax.set_ylabel(r'$g^i(\chi)$', fontsize=20)
    ax.tick_params('both', length=5, width=2, which='major', labelsize=15)
    fig.tight_layout()
    plt.savefig(figname)
    #plt.show()
    plt.close()
#-------------------------------------------------------#
def main():
    scale_n = 1.10                         # Tully-Fisher total surface number density (unit: arcmin^-2), from Eric et al.(2013), Table 2 (TF-Stage)
    c = 2.99792458e5                       # speed of light unit in km/s
    N_dset = (nrbin+1)*nrbin//2            # numbe of C^ij(l) data sets
    zbin = np.zeros(nrbin+1)               # for both zbin and chibin, the first element is 0.
    chibin = np.zeros(nrbin+1)

    inputf = '../Input_files/nz_stage_IV.txt'             # Input file of n(z) which is the galaxy number density distribution in terms of z
    # Here center_z denotes z axis of n(z). It may not be appropriate since we don't have redshift bin setting
    center_z, n_z = np.loadtxt(inputf, dtype='f8', comments='#', unpack=True)
    spl_nz = InterpolatedUnivariateSpline(center_z, n_z)
    n_sum = spl_nz.integral(center_z[0], center_z[-1])                      # Calculate the total number density
    #print(n_sum)
    scale_dndz = scale_n/n_sum
    n_z = n_z * scale_dndz                                                  # rescale n(z) to match the total number density from the data file equal to scale_n
    spl_nz = InterpolatedUnivariateSpline(center_z, n_z)                    # Interpolate n(z) in terms of z using spline

    # bin interval
    z_min = center_z[0]
    z_max = 2.0   # based on the data file, at z=2.0, n(z) is very small
    zbin_avg = (z_max-z_min)/float(nrbin)
    for i in range(nrbin):
        zbin[i]=i*zbin_avg + z_min
        if zbin[i] <= z_target:
            target_i = i
    zbin[-1]= z_max

    # Note that here chibin[0] is not equal to 0, since there is redshift cut at low z.
    for i in range(0, nrbin+1):
        ##chibin[i] = comove_d(zbin[i])*c/100.0
        # here we just ignore the constant c/100.0, as we just show len efficiency which is unitless
        chibin[i] = comove_d(zbin[i])

    ##print('Xmax', c/100.0*comove_d(zbin[-1]))
    Xmax = comove_d(zbin[-1])
    print('Xmax')

    print('target_i:', target_i)
    z_array = np.linspace(0.0, zbin[target_i+1], 1000, endpoint=False)
    chi_array = np.array([comove_d(z_i) for z_i in z_array])
    print(chi_array)
    g_i_array = np.array([], dtype=np.float64)
    for chi_k in chi_array:
        g_i = lens_eff(chibin, target_i, chi_k)
        g_i_array = np.append(g_i_array, g_i)

    odir = '/Users/ding/Documents/playground/shear_ps/project_final/fig_lens_eff/gi_data/'
    if not os.path.exists(odir):
        os.mkdir(odir)
    ofile = odir + 'gi_{}_nrbin{}_zk_{}_rbinid_{}.npz'.format(survey_stage, nrbin, z_target, target_i)
    np.savez(ofile, z=z_array, gi=g_i_array)

    odir = './figs/lens_eff_gi/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    ofile = odir + 'gi_{}_nrbin{}_zi_{}.pdf'.format(survey_stage, nrbin, z_target)
    plot_lens_eff(z_array, g_i_array, nrbin, z_target, ofile)



if __name__ == '__main__':
    main()

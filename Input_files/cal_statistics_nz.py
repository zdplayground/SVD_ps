#!//Users/ding/anaconda3/bin/python
# Calculate some statistic quantities of galaxy number distribution n(z). --06/08/2018
#
import numpy as np
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
import argparse

def integrand_nz(z, spl_nz):
    return spl_nz(z)

def integrand_znz(z, spl_nz):
    return spl_nz(z)*z

def main():
    parser = argparse.ArgumentParser(description='Calculate z_mean and z_median of a given n(z) distribution.')
    parser.add_argument("--survey_stage", help="The stage of survey, either stage_III or stage_IV.", required=True)

    args = parser.parse_args()
    survey_stage = args.survey_stage

    ifile_dict = {'stage_III': 'zdistribution_DES_Tully_Fisher.txt',
                  'stage_IV': 'nz_stage_IV.txt'}
    zmax_dict = {'stage_III': 1.3, 'stage_IV': 2.0}

    if 'stage_III' in survey_stage:
        z, nz = np.loadtxt(ifile_dict['stage_III'], dtype='f8', comments='#', usecols=(1, 3), unpack=True)
    else:
        z, nz = np.loadtxt(ifile_dict['stage_IV'], dtype='f8', comments='#', unpack=True)

    spl_nz = InterpolatedUnivariateSpline(z, nz)
    print('spl_nz integral:', spl_nz.integral(z[0], zmax_dict[survey_stage]))

    inte_nz = integrate.quad(integrand_nz, z[0], zmax_dict[survey_stage], args=(spl_nz,), epsabs=1.e-3, epsrel=1.e-3)[0]
    inte_znz = integrate.quad(integrand_znz, z[0], zmax_dict[survey_stage], args=(spl_nz,), epsabs=1.e-3, epsrel=1.e-3)[0]
    z_mean = inte_znz/inte_nz
    print('inte_nz:', inte_nz, 'inte_znz:', inte_znz, 'z_mean:', z_mean)

    z_array = np.linspace(z[0]+0.01, zmax_dict[survey_stage], num=1000)
    inte_nz_array = np.array([], dtype=np.float64)
    for zmax in z_array:
        inte_nz_sample = integrate.quad(integrand_nz, z[0], zmax, args=(spl_nz,), epsabs=1.e-3, epsrel=1.e-3)[0]
        inte_nz_array = np.append(inte_nz_array, inte_nz_sample)

    spl_z_inte_nz = InterpolatedUnivariateSpline(inte_nz_array, z_array)
    z_median = spl_z_inte_nz(inte_nz/2.0)
    print('z_median', z_median, 'sum nz at z_med is', spl_nz.integral(z[0], z_median))

if __name__ == '__main__':
    main()

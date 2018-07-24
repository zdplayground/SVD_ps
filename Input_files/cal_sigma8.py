#!/Users/ding/miniconda3/bin/python
# Calculate sigma_8 from power spectrum. --02/20/2018
# sigma^2(R) = 1/(2 \pi^2) \int dk/k * k^3*P(k) * |W(k)|^2
#
import numpy as np
from scipy import interpolate, integrate
import matplotlib.pyplot as plt

# The corresponding window function in Fourier space from top-hat window in real space.
def Window(x):
    return 3.0*(np.sin(x)-x*np.cos(x))/x**3.0

def sigma2_integrand(k, spl_Pk, R):
    return k*k* spl_Pk(k) *(Window(k*R))**2.0

def main():
    ifile = './CAMB_Planck2015_matterpower.dat'
    k, Pk = np.loadtxt(ifile, dtype='f8', comments='#', unpack=True)
    # plt.loglog(k, Pk)
    # plt.show()
    spl_Pk = interpolate.UnivariateSpline(k, Pk)
    R = 8
    temp = integrate.quad(sigma2_integrand, k[0], k[-1], args=(spl_Pk, R), epsabs=1.e-5, epsrel=1.e-5)[0]
    sigma2_8 = temp/(2*np.pi**2.0)
    print('sigma2_8:', sigma2_8, 'sigma_8:', sigma2_8**0.5)


if __name__ == '__main__':
    main()

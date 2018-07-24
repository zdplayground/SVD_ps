# Compare n(z) in stage III and stage IV. -- 11/14/2017
#
import os
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

ifile = '../Input_files/nz_stage_IV.txt'
z_4, nz_4 = np.loadtxt(ifile, dtype='f8', comments='#', unpack=True)
tck_nz4 = interpolate.splrep(z_4, nz_4)
nz4_sum = interpolate.splint(z_4[0], z_4[-1], tck_nz4)
nz4_scale = 1.1/nz4_sum
nz_4 = nz_4 * nz4_scale

ifile = '../Input_files/zdistribution_DES_Tully_Fisher.txt'
z_3, nz_3 = np.loadtxt(ifile, dtype='f8', comments='#', usecols=(1, 3), unpack=True)
tck_nz3 = interpolate.splrep(z_3, nz_3)
nz3_sum = interpolate.splint(z_3[0], z_3[-1], tck_nz3)
nz3_scale = 1.1/nz3_sum
nz_3 = nz_3 * nz3_scale

def nz_chang(z):
    # from case k=1.0 for LSST (stage IV)
    alpha, z_0, beta = 1.24, 0.51, 1.01  # k=1
    #alpha, z_0, beta = 1.23, 0.59, 1.05  # k=2
    #alpha, z_0, beta = 1.28, 0.41, 0.97   # k=0.5
    nz = z**alpha * np.exp(-(z/z_0)**beta)
    return nz

odir = './figs/'
if not os.path.exists(odir):
    os.makedirs(odir)

figname = odir + "nz_stageIII_IV.pdf"
ofile = odir + figname
fig, ax = plt.subplots()
ax.plot(z_3, nz_3, '-', label="Stage III")
ax.plot(z_4, nz_4, '--', label="Stage IV")
#ax.plot(z_4, nz_chang(z_4)*nz4_scale, '-.')
ax.set_xlabel(r"$z$", fontsize=18)
ax.set_ylabel('$n(z)$ $[\mathtt{arcmin}]^{-2}$', fontsize=18)
ax.legend(loc='upper right', fontsize=18)
plt.tight_layout()
plt.savefig(figname)
plt.show()
plt.close()

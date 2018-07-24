#!/bin/bash
#nrbin=100
nrbin=30
nkout=66
mpirun -n 4 ./mpi_KW_cross_convergence_ps_bin.py --nrbin $nrbin --nkout $nkout --Pk_type Pwig_nonlinear --Psm_type Pnorm --cal_sn True --cal_cijl True --cal_Gm True

mpirun -n 4 ./mpi_KW_cross_convergence_ps_bin.py --nrbin $nrbin --nkout $nkout --Pk_type Pnow --Psm_type Pnorm --cal_cijl True

#mpirun -n 4 ./mpi_TF_sn_dep_Cov_cij_cross_bin.py --nrbin $nrbin --num_kout $nkout --snf 1.0 --Pk_type Pnow --idir0 ./ --odir0 ./

#mpirun -n 4 ./mpi_TF_sn_dep_Cov_cij_cross_bin.py --nrbin $nrbin --num_kout $nkout --snf 1.0 --Pk_type Pwig_nonlinear --idir0 ./ --odir0 ./

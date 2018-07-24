#!/bin/bash -l

nkout=66
snf=3.1

for nrbin in 5; do
#mpirun -n 4 ./f2py_TW_zextend_multibin.py --nrbin $nrbin --nkout $nkout --Pk_type Pwig_nonlinear --cal_sn True --cal_cijl True --cal_Gm True
#mpirun -n 4 ./f2py_TW_zextend_multibin.py --nrbin $nrbin --nkout $nkout --Pk_type Pnow --cal_cijl True

mpirun -n 4 ./mpi_PW_sn_dep_Cov_multibin.py --snf $snf --nbin_case 0 --num_kout $nkout --Pk_type Pwig_nonlinear --comm_size 4
mpirun -n 4 ./mpi_PW_sn_dep_Cov_multibin.py --snf $snf --nbin_case 0 --num_kout $nkout --Pk_type Pnow --comm_size 4
done

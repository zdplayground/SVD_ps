#!/bin/bash -l
nkout=66

for nrbin in $(seq 70 10 80); do
    echo $nrbin
    mpirun -n 4 ./mpi_KW_cross_convergence_ps_bin.py --nrbin $nrbin --nkout $nkout --Pk_type Pwig_nonlinear --Psm_type Pnorm --cal_sn True --cal_cijl True --cal_Gm True --odir0 ./
    mpirun -n 4 ./mpi_KW_cross_convergence_ps_bin.py --nrbin $nrbin --nkout $nkout --Pk_type Pnow --Psm_type Pnorm --cal_sn True --cal_cijl True --cal_Gm False --odir0 ./   
done



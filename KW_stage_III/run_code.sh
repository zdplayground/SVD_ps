#!/bin/bash -l

nkout=66
alpha=1.0
idir0_Cijl="./BAO_alpha_$alpha/"
idir0_Gm="./BAO_alpha_$alpha/"
odir0="./BAO_alpha_$alpha/"
#snf=1.0

cal_cijl="False"
cal_Gm="False"
cal_sn="False"
for nrbin in 30; do
    time mpirun -n 4 ./cal_Cijl_Gm.py --nrbin $nrbin --nkout $nkout --Pk_type Pwig_nonlinear --Psm_type Pnorm --alpha $alpha --cal_sn $cal_sn --cal_cijl $cal_cijl --cal_Gm $cal_Gm --odir0 $odir0 --show_nz True
#    time mpirun -n 4 ./cal_Cijl_Gm.py --nrbin $nrbin --nkout $nkout --Pk_type Pnow --Psm_type Pnorm --alpha $alpha --cal_sn True --cal_cijl $cal_cijl --cal_Gm False --odir0 $odir0
#for snf in 1.0; do
#    mpirun -n 4 ./mpi_KW_sn_dep_Cov_cij_cross_bin.py --nrbin $nrbin --num_kout $nkout --snf $snf --Pk_type Pwig_nonlinear --idir0_Cijl $idir0_Cijl --idir0_Gm $idir0_Gm --odir0 $odir0
#    mpirun -n 4 ./mpi_KW_sn_dep_Cov_cij_cross_bin.py --nrbin $nrbin --num_kout $nkout --snf $snf --Pk_type Pnow --idir0_Cijl $idir0_Cijl --idir0_Gm $idir0_Gm --odir0 $odir0
#done
done


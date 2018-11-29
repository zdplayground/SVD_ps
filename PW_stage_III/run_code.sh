#!/bin/bash -l
nkout=66
#nb_case_id=(4 6 8 9 10)
nrbin=30       # modify the nbin_case correspondingly
#snf=1.0
snf=0.3226    # match the effective shape noise with that of PWL-Stage IV  

cal_sn="False"
cal_cijl="False"
cal_Gm="False"

#for nrbin in 30; do
#echo $nrbin
#for alpha in 1.0; do
mpirun -n 4 ./cal_Cijl_Gm.py --nrbin $nrbin --nkout $nkout --Pk_type Pwig_nonlinear --cal_sn $cal_sn --cal_cijl $cal_cijl --cal_Gm $cal_Gm --odir0 ./ --alpha $alpha --show_nz True
#mpirun -n 4 ./cal_Cijl_Gm.py --nrbin $nrbin --nkout $nkout --Pk_type Pnow --cal_sn False --cal_cijl True --cal_Gm False --odir0 ./ --alpha $alpha
#done
#done

alpha=1.0
# be careful to set nbin_case correctly corresponding to nrbin value.
for nrbin_case in 1; do
mpirun -n 4 ./mpi_PW_sn_dep_Cov_multibin.py --snf $snf --nbin_case $nrbin_case --num_kout $nkout --Pk_type Pwig_nonlinear --comm_size 4 --alpha $alpha
mpirun -n 4 ./mpi_PW_sn_dep_Cov_multibin.py --snf $snf --nbin_case $nrbin_case --num_kout $nkout --Pk_type Pnow --comm_size 4 --alpha $alpha
done
   

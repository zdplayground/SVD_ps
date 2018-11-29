#!/bin/bash -l
# be careful to set nbin_case correctly corresponding to nrbin value.
nkout=66
nbin_list=(5 6 7 8 9 10 15 20 22 25 27 30 32 35 37) 
#nb_case_id=(4 6 8 9 10)
nrbin=30       # modify the nbin_case correspondingly
#snf=1.0
snf=0.18385     # match the effective shape noise of PW-Stage IV with that of KW-Stage IV 

#count=0
#for nrbin in 6 7 8 9; do
#echo $nrbin
#for alpha in 1.0; do
#mpirun -n 4 ./f2py_TW_zextend_multibin.py --nrbin $nrbin --nkout $nkout --Pk_type Pwig_nonlinear --cal_sn True --cal_cijl True --cal_Gm True --odir0 ./ --alpha $alpha
#mpirun -n 4 ./f2py_TW_zextend_multibin.py --nrbin $nrbin --nkout $nkout --Pk_type Pnow --cal_sn False --cal_cijl True --cal_Gm False --odir0 ./ --alpha $alpha
#done
#done

alpha=1.0
##nbin_case=${nb_case_id[$count]}
for nbin_case in 1 2 3 4; do
echo $nbin_case
##for snf in 1.8385e-06 1.8385e-05 1.8385e-04 1.8385e-03; do
mpirun -n 4 ./mpi_PW_sn_dep_Cov_multibin.py --snf $snf --nbin_case $nbin_case --num_kout $nkout --Pk_type Pwig_nonlinear --comm_size 4 --alpha $alpha
mpirun -n 4 ./mpi_PW_sn_dep_Cov_multibin.py --snf $snf --nbin_case $nbin_case --num_kout $nkout --Pk_type Pnow --comm_size 4 --alpha $alpha
done
  

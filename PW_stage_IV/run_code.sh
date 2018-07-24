#!/bin/bash -l
# be careful to set nbin_case correctly corresponding to nrbin value.
nkout=66
nbin_list=(5 10 15 20 22 25 27 30 32 35 37) 
#nb_case_id=(4 6 8 9 10)
nrbin=30
snf=1.0
#snf=0.1838    # match the effective shape noise of PW-Stage IV with that of KW-Stage IV 

count=0
#for nrbin in ${nbin_list[*]}; do
echo $nrbin
#mpirun -n 4 ./f2py_TW_zextend_multibin.py --nrbin $nrbin --nkout $nkout --Pk_type Pwig_nonlinear --cal_sn False --cal_cijl True --cal_Gm True 
#mpirun -n 4 ./f2py_TW_zextend_multibin.py --nrbin $nrbin --nkout $nkout --Pk_type Pnow --cal_sn False --cal_cijl True --cal_Gm False
#nbin_case=${nb_case_id[$count]}
#nbin_case=$count
nbin_case=7
echo $nbin_case
mpirun -n 4 ./mpi_PW_sn_dep_Cov_multibin.py --snf $snf --nbin_case $nbin_case --num_kout $nkout --Pk_type Pwig_nonlinear --comm_size 4
#mpirun -n 4 ./mpi_PW_sn_dep_Cov_multibin.py --snf $snf --nbin_case $nbin_case --num_kout $nkout --Pk_type Pnow --comm_size 4
    
let "count += 1"
#done   

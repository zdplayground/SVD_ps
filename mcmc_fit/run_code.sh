#!/bin/bash -l
#sleep 25200
lt=TF
survey_name=KW_stage_III
nrbin=30
nkbin=66
snf=1.0
#snf=3.1    # for PW_stage_III
#snf=0.1838     # match the effective shape noise of PW-Stage IV with the default KW-Stage IV 
#snf=0.2558    # the effective shape noise of optimistic KW-Stage IV

#survey_stage=$survey_name/precise_Cijl_Gm/nersc_env/apply_SVc_on_Pk_and_CovP
survey_stage=$survey_name/precise_Cijl_Gm/apply_SVc_on_Pk_and_CovP
#survey_stage=$survey_name/apply_SVc_on_Pk_and_CovP

SVc_on_CovP=True
#start_lmin=4        # for Stage IV
start_lmin=10      # for Stage III

nrbin_list=(5 10 15 20 22 25 27 32 35 37)
#for nrbin in ${nrbin_list[*]}; do
#for nrbin in $(seq 100 10 100);do
echo $nrbin
mpirun -n 4 ./mcmc_fit_Pwig_over_Pnow.py --lt $lt --nrbin $nrbin --nkbin $nkbin --start_lmin $start_lmin --shapenf $snf --kmin 0.015 --kmax 0.3 --params_str 101 --Pwig_type Pwig_nonlinear --survey_stage $survey_stage --set_SVc_on_CovP $SVc_on_CovP  
#for alpha in $(seq 0.80 0.05 1.21); do
#mpirun -n 4 ./mcmc_fit_Pwig_over_Pnow.py --lt $lt --nrbin $nrbin --nkbin $nkbin --start_lmin $start_lmin --shapenf $snf --kmin 0.015 --kmax 0.3 --params_str 001 --alpha $alpha --Pwig_type Pwig_nonlinear --survey_stage $survey_stage --set_SVc_on_CovP $SVc_on_CovP  

#mpirun -n 4 ./mcmc_fit_Pwig_over_Pnow.py --lt $lt --nrbin $nrbin --nkbin $nkbin --start_lmin $start_lmin --shapenf $snf --kmin 0.015 --kmax 0.3 --params_str 001 --Sigma2_inf True --alpha $alpha --Pwig_type Pwig_nonlinear --survey_stage $survey_stage --set_SVc_on_CovP $SVc_on_CovP  

#done

#!/bin/bash -l
lt=TW
survey_name=PW_stage_III
nrbin=30
nkbin=66

#snf=1.0
snf=0.3226       # for PW_stage_III, matching the effective shape noise with that of PWL=Stage IV
#snf=0.18385     # match the effective shape noise of PW-Stage IV with the default KW-Stage IV 
#snf=0.2558    # the effective shape noise of optimistic KW-Stage IV
#snf=0.0512     # decrease shape noise 5 times from the optimistic KW-Stage IV
#snf=1e-06
alpha=1.0
params_str="101"

survey_stage=$survey_name/BAO_alpha_$alpha/apply_SVc_on_Pk_and_CovP
#survey_stage=$survey_name/BAO_alpha_$alpha/apply_SVc_on_Pk_and_CovP/nersc_env
##survey_stage=$survey_name/BAO_alpha_$alpha/apply_SVc_on_CovP    # only apply SVc on CovP but not on P(k)

SVc_on_CovP=True
#start_lmin=4        # for Stage IV
#start_lmin=46
start_lmin=10      # for Stage III
save_sampler="False"
#save_sampler="True"
modify_Cov_cij_cpq='False'

#nrbin_list=(5 10 15 20 22 25 27 32 35 37)
for nrbin in 30; do
#echo $nrbin
##for snf in 1.8385e-06 1.8385e-05 0.00018385; do
##for snf in 1.0; do
mpirun -n 4 ./mcmc_fit_Pwig_over_Pnow.py --lt $lt --nrbin $nrbin --nkbin $nkbin --start_lmin $start_lmin --shapenf $snf --kmin 0.015 --kmax 0.3 --params_str $params_str --Pwig_type Pwig_nonlinear --survey_stage $survey_stage --set_SVc_on_CovP $SVc_on_CovP --nSV_min 1 --save_sampler $save_sampler --modify_Cov_cij_cpq $modify_Cov_cij_cpq
##done
#done

#------- This block is to measure the significance of BAO detection.-------- 
#for alpha in $(seq 0.80 0.05 1.21); do
#mpirun -n 4 ./mcmc_fit_Pwig_over_Pnow.py --lt $lt --nrbin $nrbin --nkbin $nkbin --start_lmin $start_lmin --shapenf $snf --kmin 0.015 --kmax 0.3 --params_str 001 --alpha $alpha --Pwig_type Pwig_nonlinear --survey_stage $survey_stage --set_SVc_on_CovP $SVc_on_CovP  
#

mpirun -n 4 ./mcmc_fit_noBAO.py --lt $lt --nrbin $nrbin --nkbin $nkbin --start_lmin $start_lmin --shapenf $snf --kmin 0.015 --kmax 0.3 --params_str 101 --Sigma2_inf True --alpha $alpha --Pwig_type Pwig_nonlinear --survey_stage $survey_stage --set_SVc_on_CovP $SVc_on_CovP  

done

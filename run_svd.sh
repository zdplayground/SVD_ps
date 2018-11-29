#!/bin/bash -l
# For the regenerated data files (correct lensing efficiency), we have used tried to generate precise Cijl and Gm. So no
# need to note precise_Cijl_Gm folder for $idir0 or $odir0. --09/13/2018
# 
lt=TW
survey_name=PW_stage_III
nrbin=30
nkout=66
nrank=4

#snf=1.0
snf=0.3226
#snf=0.2558
#snf=0.18385

alpha=1.0     # BAO scale shifting parameter
idir0=./$survey_name/BAO_alpha_$alpha/            
odir0=./$survey_name/BAO_alpha_$alpha/

SVc_on_Pk=True
##SVc_on_Pk=False       # only for testing
SVc_on_CovP=True
start_lmin=10            # 4 for stage IV and 10 for stage III
modify_Cov_cij_cpq=False

nrbin_list=(5 10 15 20 22 25 27 30 32 35 37)
#for nrbin in $(seq 60 10 70); do;
for nrbin in 30; do
#for snf in 1.8385e-04 1.8385e-05 1.8385e-06; do
#start_lmin=46

echo $snf
./svd_cov_p_cross_multibin.py --lt $lt --nrbin $nrbin --nkout $nkout --shapenf $snf --nrank $nrank --Psm_type Pnorm --Pk_type Pnow --output_Pk_type Pk_now --idir0 $idir0 --odir0 $odir0 --set_SVc_on_Pk $SVc_on_Pk --set_SVc_on_CovP $SVc_on_CovP --start_lmin $start_lmin --modify_Cov_cij_cpq $modify_Cov_cij_cpq

./svd_cov_p_cross_multibin.py --lt $lt --nrbin $nrbin --nkout $nkout --shapenf $snf --nrank $nrank --Psm_type Pnorm --Pk_type Pwig_nonlinear --output_Pk_type Pk_wig --idir0 $idir0 --odir0 $odir0 --set_SVc_on_Pk $SVc_on_Pk --set_SVc_on_CovP $SVc_on_CovP --start_lmin $start_lmin --modify_Cov_cij_cpq $modify_Cov_cij_cpq

./svd_cov_p_cross_multibin.py --lt $lt --nrbin $nrbin --nkout $nkout --shapenf $snf --nrank $nrank --Psm_type Pnorm --Pk_type Pwig_nonlinear --output_Pk_type Pk_wnw --idir0 $idir0 --odir0 $odir0 --set_SVc_on_Pk $SVc_on_Pk --set_SVc_on_CovP $SVc_on_CovP --start_lmin $start_lmin --modify_Cov_cij_cpq $modify_Cov_cij_cpq

done


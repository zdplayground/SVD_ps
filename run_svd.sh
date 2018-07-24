#!/bin/bash -l

lt=TF
survey_name=KW_stage_III
nrbin=30
nkout=66
nrank=4

snf=1.0
#snf=3.1
#snf=0.2558
#snf=0.1838

idir0=./$survey_name/precise_Cijl_Gm/    # for KW stage surveys
odir0=./$survey_name/precise_Cijl_Gm/
#idir0=./$survey_name/                  # for PW stage surveys
#odir0=./$survey_name/

SVc_on_Pk=True
SVc_on_CovP=True
start_lmin=10    # 4 for stage IV and 10 for stage III

nrbin_list=(5 10 15 20 22 25 27 30 32 35 37)
#for nrbin in $(seq 60 10 70); do
#for nrbin in ${nrbin_list[*]}; do
./svd_cov_p_cross_multibin.py --lt $lt --nrbin $nrbin --nkout $nkout --shapenf $snf --nrank $nrank --Psm_type Pnorm --Pk_type Pnow --output_Pk_type Pk_now --idir0 $idir0 --odir0 $odir0 --set_SVc_on_Pk $SVc_on_Pk --set_SVc_on_CovP $SVc_on_CovP --start_lmin $start_lmin

./svd_cov_p_cross_multibin.py --lt $lt --nrbin $nrbin --nkout $nkout --shapenf $snf --nrank $nrank --Psm_type Pnorm --Pk_type Pwig_nonlinear --output_Pk_type Pk_wig --idir0 $idir0 --odir0 $odir0 --set_SVc_on_Pk $SVc_on_Pk --set_SVc_on_CovP $SVc_on_CovP --start_lmin $start_lmin

./svd_cov_p_cross_multibin.py --lt $lt --nrbin $nrbin --nkout $nkout --shapenf $snf --nrank $nrank --Psm_type Pnorm --Pk_type Pwig_nonlinear --output_Pk_type Pk_wnw --idir0 $idir0 --odir0 $odir0 --set_SVc_on_Pk $SVc_on_Pk --set_SVc_on_CovP $SVc_on_CovP --start_lmin $start_lmin
#done


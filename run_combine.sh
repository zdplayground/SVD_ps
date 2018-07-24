#!/bin/bash -l

lt=TF
survey_stage="KW_stage_III/precise_Cijl_Gm/nersc_env/"
#Pk_type="Pwig_nonlinear"
Pk_type="Pnow"
nkbin=66
nrank=48
get_cijl="True"   # True or False
get_gm="False"

nzbin_list=(12 18 25 27 34 37 40 44 46)

#for nzbin in ${nzbin_list[*]}; do
for nzbin in 30; do
echo $nzbin
./combine_Cijl_Gm_bin_data.py --lt $lt --nkbin_out $nkbin --nrank $nrank --Pk_type $Pk_type --survey_stage $survey_stage --nzbin $nzbin --get_cijl $get_cijl --get_gm $get_gm 
done

#!/bin/bash -l

lt=TW
#survey_stage="KW_stage_IV/precise_Cijl_Gm/BAO_alpha_1.0002/"
survey_stage="PW_stage_III/BAO_alpha_1.002/"
Pk_type="Pwig_nonlinear"
#Pk_type="Pnow"
nkbin=66
nrank=4
get_cijl="True"   # True or False
get_gm="False"

nzbin_list=(12 18 25 27 34 37 40 44 46)

#for nzbin in ${nzbin_list[*]}; do
for nzbin in 6; do
echo $nzbin
./combine_Cijl_Gm_bin_data.py --lt $lt --nkbin_out $nkbin --nrank $nrank --Pk_type $Pk_type --survey_stage $survey_stage --nzbin $nzbin --get_cijl $get_cijl --get_gm $get_gm 
done

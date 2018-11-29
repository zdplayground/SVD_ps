#!/bin/bash -l
lt=TF
alpha=1.0
survey_stage="KW_stage_IV/BAO_alpha_$alpha"

nkbin=66
nrank=4
get_cijl="True"   # True or False

nzbin_list=(6 12 18 25 27 34 37 40 44 46)
for Pk_type in Pwig_nonlinear Pnow; do

if [ $Pk_type = "Pnow" ]; then
  get_gm="False"
else
  get_gm="True"
fi
for nzbin in 10 15; do
#for nzbin in ${nzbin_list[*]:0:1}; do
echo $nzbin
./combine_Cijl_Gm_bin_data.py --lt $lt --nkbin_out $nkbin --nrank $nrank --Pk_type $Pk_type --survey_stage $survey_stage --nzbin $nzbin --get_cijl $get_cijl --get_gm $get_gm 
done
done

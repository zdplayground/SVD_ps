#!/bin/bash
for nbin_case in $(seq 0 2); do
  echo "nbin_case: $nbin_case"
  mpirun -n 4 ./PW_SN_ratio_Takada_Jain.py --snf 1.0 --nbin_case $nbin_case --Pk_type Pwig_nonlinear --comm_size 4 --idir0 ./ --odir0 ./
done

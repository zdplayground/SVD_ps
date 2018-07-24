#!/bin/bash -l
dir0=`pwd`
nrbin=30
nkout=66

#mpirun -n 4 ./mpi_KW_cross_convergence_ps_bin.py --nrbin $nrbin --nkout $nkout --Pk_type Pwig_nonlinear --Psm_type Pnorm --cal_sn True --cal_cijl True --cal_Gm True
#mpirun -n 4 ./mpi_KW_cross_convergence_ps_bin.py --nrbin $nrbin --nkout $nkout --Pk_type Pnow --Psm_type Pnorm --cal_sn False --cal_cijl True --cal_Gm False

mpirun -n 4 ./mpi_TF_sn_dep_Cov_cij_cross_bin.py --nrbin $nrbin --num_kout $nkout --snf 1.0 --Pk_type Pwig_nonlinear --idir0_Cijl ./ --idir0_Gm ./ --odir0 ./
mpirun -n 4 ./mpi_TF_sn_dep_Cov_cij_cross_bin.py --nrbin $nrbin --num_kout $nkout --snf 1.0 --Pk_type Pnow --idir0_Cijl ./ --idir0_Gm ./ --odir0 ./

#./svd_cov_p_cross_multibin.py --lt TF --nrbin $nrbin --nkout $nkout --shapenf 1.0 --Pk_type Pwig_nonlinear --nrank 4 --Psm_type Pnorm --output_Pk_type Pk_wig --idir0 ./
#./svd_cov_p_cross_multibin.py --lt TF --nrbin $nrbin --nkout $nkout --shapenf 1.0 --Pk_type Pwig_nonlinear --nrank 4 --Psm_type Pnorm --output_Pk_type Pk_wnw --idir0 ./
#./svd_cov_p_cross_multibin.py --lt TF --nrbin $nrbin --nkout $nkout --shapenf 1.0 --Pk_type Pnow --nrank 4 --Psm_type Pnorm --output_Pk_type Pk_now --idir0 ./

#cd /Users/ding/Documents/playground/shear_ps/SVD_ps/mcmc_fit
#time mpirun -n 4 ./mcmc_fit_Pwig_over_Pnow.py --lt TF --nrbin $nrbin --nkbin $nkout --shapenf 1.0 --kmin 0.015 --kmax 0.3 --params_str 101 --Pwig_type Pwig_nonlinear --survey_stage KW_stage_III/precise_Cijl_Gm

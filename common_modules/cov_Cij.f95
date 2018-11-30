! made on 03/03/2016, define a function calculating covariance matrix of C^ij(l) including cross terms. 
! The algorithm is copied from sn_dep_Cov_cij_cross.py
! Use f2py -m cov_matrix_module -h cov_matrix_module.pyf cov_Cij.f95 --overwrite-signature to generate
! signature file.
! Use f2py -c --fcompiler=gnu95 cov_matrix_module.pyf cov_Cij.f95 to generate the module
! modified: 04/12/2016, add !f2py for subroutine cal_id_dset
!
subroutine cal_Cov_matrix(red_bin, dset_len, iu1, cij, Cov_cij_cpq)
    implicit none

    integer:: red_bin, dset_len 
    integer:: iu1(2, 0:dset_len-1)
    double precision:: cij(0:dset_len-1)
!f2py intent(in):: red_bin, dset_len, iu1, cij
!f2py intent(out):: Cov_cij_cpq

    integer:: ru,cv,i,j,p,q,u1,u2,v1,v2
    double precision, dimension(0:dset_len-1, 0:dset_len-1):: Cov_cij_cpq

    Cov_cij_cpq = 0.d0
    do ru = 0, dset_len-1
        do cv = ru, dset_len-1
            i = iu1(1, ru)
            j = iu1(2, ru)
            p = iu1(1, cv)
            q = iu1(2, cv)
            call cal_id_dset(red_bin, i, p, u1)
            call cal_id_dset(red_bin, i, q, u2)
            call cal_id_dset(red_bin, min(j, q), max(j, q), v1)
            call cal_id_dset(red_bin, min(j, p), max(j, p), v2)
            Cov_cij_cpq(ru, cv) = cij(u1)*cij(v1) + cij(u2)*cij(v2)
        end do
    end do
    return
end subroutine cal_Cov_matrix

subroutine cal_id_dset(red_bin, i, p, u1)
    implicit none
    ! using Python rule, i, p stars from 0
    integer:: red_bin, i, p, u1
!f2py    intent(in):: red_bin, i, p
!f2py    intent(out):: u1
 
    u1 = (2*red_bin-i-1) * i/2 + p

    return
end subroutine cal_id_dset
            


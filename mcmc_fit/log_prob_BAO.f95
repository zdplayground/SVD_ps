! Copy the code from /Users/ding/Documents/playground/WiggleNowiggle/HS_pre_rec_run2346/pre_log_prob_rsd_HS.f90, simplify it to calculate Pk_model. -- 08/10/2017
! The likelihood function could be referenced from Seo et al. 2012. -- 08/10/2017
! Add constant A as the amplitude parameter. -- 08/17/2017
! Change module name from lnprob_module to lnprob_linear. -- 10/31/2018
!
! Use f2py -m lnprob_linear -h lnprob_linear.pyf log_prob_BAO.f95 --overwrite-signature to generate
! signature file.
! Use f2py -c --fcompiler=gnu95 lnprob_linear.pyf log_prob_BAO.f95 to generate the module
!
subroutine match_params(theta, params_indices, fix_params, params_array, dim_theta, dim_params)
    implicit none
    integer:: dim_theta, dim_params, count, i
    double precision:: theta(dim_theta), params_indices(dim_params), fix_params(dim_params)
!f2py intent(in):: theta, params_indices, fix_params
    double precision:: params_array(dim_params)
!f2py intent(out):: params_array

    count = 1  ! Be very careful that the starting value is different from Python's. It's 1 in fortran!
    do i=1, dim_params
        if (params_indices(i) == 1) then
            params_array(i) = theta(count)
            count = count + 1
        else
            params_array(i) = fix_params(i)
        endif
    end do
    return

end subroutine match_params

subroutine cal_Pk_model(Pk_linw, Pk_sm, k_t, sigma2, a, Pk_model, dim_kt)
    implicit none
    integer:: dim_kt, i
    double precision, dimension(dim_kt):: Pk_linw, Pk_sm, k_t, Pk_model
    double precision:: sigma2, a
!f2py intent(in):: Pk_linw, Pk_sm, k_t, sigma2, a
!f2py intent(out):: Pk_model
    do i=1, dim_kt
        Pk_model(i) = a * (1.d0 + (Pk_linw(i)/Pk_sm(i) - 1.d0)*exp(-k_t(i)**2.0 * sigma2/2.0))
    enddo
    return
end subroutine cal_Pk_model

subroutine lnprior(theta, params_indices, fix_params, lp, dim_theta, dim_params)
    implicit none
    integer:: dim_theta, dim_params
    double precision, dimension(dim_theta):: theta
    double precision, dimension(dim_params):: params_indices, fix_params, params_array
    double precision:: lp
    double precision:: alpha, sigma2, A
!f2py intent(in):: theta, params_indices, fix_params
!f2py intent(out):: lp
    call match_params(theta, params_indices, fix_params, params_array, dim_theta, dim_params)
    alpha = params_array(1)
    sigma2 = params_array(2)
    A = params_array(3)

    if (alpha > 0.5d0 .and. alpha<1.5d0 .and. A > 0.d0 .and. A<1.5d0) then
        lp = 0.d0
    else
        lp = -1.d30  ! return a negative infinitely large number
    endif
    !print*, lp
    return
end subroutine lnprior

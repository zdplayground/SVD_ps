!--------------------------------------
! lens function for finite integration of G', made on 05/10/2018
! Use f2py -m lens_eff_module -h lens_eff_module.pyf lens_eff_fun.f95 --overwrite-signature to generate
! signature file.
! Use f2py -c --fcompiler=gnu95 lens_eff_module.pyf lens_eff_fun.f95 to generate the module
!--------------------------------------
subroutine lens_eff(i, nrbin, chi_k, chibin, eps, g_i)
    implicit none
    integer:: i, nrbin
    double precision, dimension(0:nrbin):: chibin
    double precision:: chi_k, x_low, x_up, x_diff, eps, g_i
!f2py intent(in):: i, nrbin, chi_k, chibin, eps
!f2py intent(out):: g_i 

    x_low = max(chibin(i), chi_k)
    x_up = chibin(i+1)
    x_diff = x_up - x_low
    if (x_diff > eps) then
        g_i = 1.d0 - (log(x_up) - log(x_low))/x_diff * chi_k
    else
        !g_i = 1.0 - chi_k/x_low  ! or g_i = 0.d0      because it's close to 0. 
        g_i = 0.d0
    endif

end subroutine lens_eff


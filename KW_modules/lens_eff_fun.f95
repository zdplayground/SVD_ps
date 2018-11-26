!--------------------------------------
! lens function for finite integration of G', made on 05/10/2018
! 1. Modify x_diff to be chi(i+1)-chi(i), which is effect of normalization of n^i(z) over all z range.
! 2. Remove eps parameter. No need to judge how close chi_k goes to chi_{i+1}.  --09/12/2018
! 3. Input chi_k should be less than chibin(i+1)
!
! Use f2py -m lens_eff_module -h lens_eff_module.pyf lens_eff_fun.f95 --overwrite-signature to generate
! signature file.
! Use f2py -c --fcompiler=gnu95 lens_eff_module.pyf lens_eff_fun.f95 to generate the module
!--------------------------------------
subroutine lens_eff(i, nrbin, chi_k, chibin, g_i)
    implicit none
    integer:: i, nrbin
    double precision, dimension(0:nrbin):: chibin
    double precision:: chi_k, x_low, x_up, x_diff, d_x_uplow, g_i
!f2py intent(in):: i, nrbin, chi_k, chibin
!f2py intent(out):: g_i

    x_low = max(chibin(i), chi_k)
    x_up = chibin(i+1)
    x_diff = chibin(i+1) - chibin(i)
    d_x_uplow = x_up - x_low
    g_i = (d_x_uplow - chi_k * log(x_up/x_low))/x_diff

end subroutine lens_eff

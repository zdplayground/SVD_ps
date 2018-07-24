subroutine comoving_d(z, cdis)
    implicit none
    integer:: num_abs, i
    parameter(num_abs=30)
    double precision:: z, cdis
!f2py    intent(in):: z
!f2py    intent(out):: cdis
    double precision:: x1, x2, x(num_abs), w(num_abs)
    double precision:: zp, omega_m, omega_L
    parameter(omega_m = 0.3175)
    parameter(omega_L = 0.6825)
 
    x1 = 0.d0
    x2 = z
    cdis = 0.d0

    call gauleg(x1, x2, x, w, num_abs)
    do i=1, num_abs
        zp = x(i)
        cdis = cdis+ w(i)*1.0/(omega_m*(1.d0+zp)**3.d0+omega_L)
    enddo

    return
end subroutine comoving_d


    

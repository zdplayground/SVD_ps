! Copy the code from wrong_lens_efficiency/PW_stage_IV. --09/05/2018
! ------------------------------------------------------------------------------------
! 1. Test the x value of erf(x). Make it (variable const in the function lens_eff) larger to have 5sigma confidence, which should not
! differ much from the results from 3sigma confidence. --07/10/2018
! 2. Modify lensing efficiency. photo-z galaxy number density distribution from each tomographic bin should be normalized. --08/31/2018
!
! Use f2py -m lens_eff_module -h lens_eff_module.pyf lens_eff_fun.f95 comove_dis.f95 gauleg.f95 spline.f95 --overwrite-signature to generate
! signature file.
! Use f2py -c --fcompiler=gnu95 lens_eff_module.pyf lens_eff_fun.f95 comove_dis.f95 gauleg.f95 spline.f95 to generate the module

subroutine lens_eff(num_bin, zbin, num_z, center_z, n_z, nz_y2, rb_id, z_k, sigma_z_const, g_i)
! num_bin: number of redshift bins
! zbin: stores boundary points of num_bin redshift bins, zbin(1)=0.0, zbin(num_bin+1)=z_max
! num_z: redshift points from input data file, e.g. 'zdistribution_DES_Tully_Fisher.txt' for PW-Stage III
! center_z: redshift z from data file, using the second column
! n_z: galaxy number distribution(spectroscopic) n(z) in terms of z
! nz_y2: the second derivative of n(z), for spline and splint routines
! rb_id: tomographic bin id, min=0, max=num_bin-1
! z_k: the lower integration boundary point
! sigma_z_const: the constant of sigma_z (systematic error of source redshift from photometry)
! g_i: output of the function lens_eff
    implicit none
    integer:: num_bin, num_z, rb_id
    double precision, dimension(num_z):: center_z, n_z, nz_y2
    double precision:: z_k, zbin(0:num_bin), sigma_z_const, g_i
!f2py    intent(in):: num_bin, zbin, num_z, center_z, n_z, nz_y2, rb_id, z_k, sigma_z_const
!f2py    intent(out):: g_i

    integer:: num_abs,i
    parameter(num_abs=50)            !Fron num_abs=50 to 80, answer converges,precision reaches 10^-5
! x1, x2 are initial and final boundaries of integration, respectively
    double precision:: diff_zphz, x1, x2, x(num_abs), w(num_abs), x_p2(num_abs), w_p2(num_abs)
    double precision:: zp, ni_zp, cdis_zk, cdis_zp, erf_z, zlb, zub, const, gi_p1, gi_p2

    !rb_id starts from 0, maximum is num_bin-1
    diff_zphz = z_k-zbin(rb_id+1)    !difference between z_k and up boundary of ith redshift bin
    const = 2.1d0 * 2.0**0.5 * sigma_z_const
    !const = 3.5d0 * 2.0**0.5 * sigma_z_const  ! difference from the above is ~10^-5
    erf_z = const*(1.+z_k)           !2.1 factor guarantee 3 sigma precision, 3.5 factor gives ~5 sigma precision
!(erf_z considers up boundary for integration, we can set low boundary for calculation efficiency)
    g_i = 0.d0
    if( diff_zphz < erf_z )then
        gi_p1 = 0.d0
        gi_p2 = 0.d0
! set the low boundary of integration as x1
        zlb = (zbin(rb_id)- const)/(1.+ const) !0.1485 =2.1*sqrt(0.5)*0.05
        zub = (zbin(rb_id+1)+const)/(1.-const)
        x1 = max(zlb, z_k)
        x2 = min(zub, zbin(num_bin))
        call comoving_d(z_k, cdis_zk)          !calculate chi_zk
        call gauleg(x1, x2, x, w, num_abs)

        call gauleg(zlb, x2, x_p2, w_p2, num_abs)
        ! compared with the above case, the relative error is 0.1% (using 3sigma in const)
        !call gauleg(0.d0, zbin(num_bin), x_p2, w_p2, num_abs)

        do i=1, num_abs
           zp = x(i) !z^prime
           call photo_ni(num_bin, zbin, num_z, center_z, n_z, nz_y2, rb_id, zp, sigma_z_const, ni_zp)   !get ni_sz
           !get chi(z)
           call comoving_d(zp, cdis_zp)
           gi_p1 = gi_p1 + ni_zp * (1.d0 - cdis_zk/cdis_zp) * w(i)

           zp = x_p2(i)
           call photo_ni(num_bin, zbin, num_z, center_z, n_z, nz_y2, rb_id, zp, sigma_z_const, ni_zp)
           gi_p2 = gi_p2 + ni_zp * w_p2(i)
        enddo

        g_i = gi_p1 / gi_p2        ! n^i(z) should be normalized over all z range
    endif

end subroutine lens_eff

! subroutine phot_ni output n_i(z) from ith photo-z bin. z denotes spectroscopic z.
subroutine photo_ni(num_bin, zbin, num_z, center_z, n_z, nz_y2, rb_id, s_z, sigma_z_const, ni_sz)
! num_bin: number of redshift bins
! zbin: stores boundary points of num_bin redshift bins, zbin(1)=0.0, zbin(num_bin+1)=z_max
! num_z: redshift points from the data file 'zdistribution_DES_Tully_Fisher.txt'
! center_z: redshift z from data file, using the second column
! n_z: galaxy number distribution n(z) in terms of z
! nz_y2: the second derivative of n(z), for spline and splint routines
! rb_id: tomographic bin id
! s_z: input source redshift, as an integration abscissa
! sigma_z_const: the constant of sigma_z (systematic error of source redshift from photometry)
! ni_sz: output photometric number density in terms of s_z
    implicit none
    integer:: num_bin, num_z, rb_id
!f2py    intent(in):: num_bin, num_z, rb_id
    double precision,dimension(num_z)::center_z, n_z, nz_y2
    double precision::s_z, ni_sz, zbin(0:num_bin), sigma_z_const
!f2py    intent(in):: center_z, n_z, nz_y2, s_z, zbin, sigma_z_const
!f2py    intent(out):: ni_sz
    double precision:: sigma_sz, erfx_up, erfx_low,var_up,var_low
    double precision:: n_sz !splint value of n(z)

    sigma_sz = sigma_z_const*(1.+s_z)
    erfx_up = (zbin(rb_id+1)-s_z)/(2.d0**0.5 * sigma_sz)
    erfx_low = (zbin(rb_id)-s_z)/(2.d0**0.5 * sigma_sz)
    var_up = erf(erfx_up)
    var_low = erf(erfx_low)
    call splint(center_z, n_z, nz_y2, num_z, s_z, n_sz) !use spline interpolation to get n_sz at s_z
    ni_sz = (var_up-var_low) * 0.5 *n_sz
    return

end subroutine photo_ni
